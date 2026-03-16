import os
import subprocess
import torch
import yaml
import sys
import time
import argparse
import glob

TRAIN_COMMAND = "openrlhf.cli.train_dpo"

def find_next_open_port(start_port=19500, max_port=65535):
    import socket
    """Finds the next available open port starting from `start_port`."""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port  # Found an open port
            except OSError:
                continue  # Port is in use, try the next one
    raise RuntimeError("No available ports found in the specified range.")

def parse_args():
    parser = argparse.ArgumentParser(description="Launch DPO training with specified configuration.")
    parser.add_argument('--openrlhf_dir', type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), help='Path to the OpenRLHF directory (defaults to repo root)')
    parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
    parser.add_argument('--deepspeed_stage', type=int, choices=[0, 1, 2, 3], default=None, help='DeepSpeed stage')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs to use')
    parser.add_argument('--train_yaml_path', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--train_overrides', type=str, default=None, help='Overrides for the training config')

    # split up the configs across multiple jobs
    parser.add_argument('--num_jobs', type=int, default=1, help='Number of jobs to split the configs across')
    parser.add_argument('--job_idx', type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID", 0), help='Index of the current job')

    args = parser.parse_args()
    return args

def launch_training_pipeline(args, config):
    model_path = os.path.join(args.openrlhf_dir, config['save_path'])

    curr_dir = os.getcwd()
    os.chdir(args.openrlhf_dir)

    print(f"Training model using {args.num_gpus} GPUs")
    print(f"Training config: {args.train_yaml_path}")
    print(f"DeepSpeed stage: {args.deepspeed_stage}")

    cuda_visible_devices = None
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        print(f"Using CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        del os.environ['CUDA_VISIBLE_DEVICES']

    if args.deepspeed_stage is not None:
        config["zero_stage"] = args.deepspeed_stage

    if args.train_overrides is not None:
        for override in args.train_overrides.split(","):
            if '=' in override:
                key, value = override.split('=')
                if value.lower() == "true":
                    config[key] = True
                elif value.lower() == "false" and key in config:
                    del config[key]
                else:
                    config[key] = value
            else:
                config[override] = True

    training_args = []
    for k, v in config.items():
        # Skip the keys that have false values
        if v is False:
            continue
        training_args.append(f"--{k}")
        # if its a --store_true argument, we don't need to add the value
        if v is True or v is None:
            continue
        # otherwise, add the value
        training_args.append(str(v))

    if args.master_port == 0:
        args.master_port = find_next_open_port()

    training_command = ['deepspeed', '--master_port', str(args.master_port)]
    if cuda_visible_devices is not None:
        training_command.extend(['--include', 'localhost:' + cuda_visible_devices])

    training_command += ['--module', TRAIN_COMMAND]
    training_command += training_args

    print(f"Running command: {' '.join(training_command)}")
    subprocess.run(training_command, env=os.environ, stdout=sys.stdout, stderr=subprocess.STDOUT)

    if cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    os.chdir(curr_dir)

    return os.path.abspath(model_path)


def main(args):
    avail_gpus = torch.cuda.device_count()
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', ",".join(map(str, range(avail_gpus))))

    if avail_gpus < args.num_gpus:
        raise ValueError(f"Number of GPUs requested ({args.num_gpus}) is greater than available GPUs ({avail_gpus})")
    elif avail_gpus > args.num_gpus:
        cuda_visible_devices = ",".join(cuda_visible_devices.split(",")[:args.num_gpus])
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    with open(args.train_yaml_path, 'r') as f:
        train_config = yaml.safe_load(f)

    starttime = time.time()
    launch_training_pipeline(args, train_config)
    print(f"Training took {time.time() - starttime} seconds")


if __name__ == "__main__":
    args = parse_args()
    if os.path.isdir(args.train_yaml_path):
        all_train_yamls = glob.glob(f"{args.train_yaml_path}/**/*.yaml", recursive=True)
    else:
        all_train_yamls = [args.train_yaml_path]

    print(f"Found {len(all_train_yamls)} training configs: {all_train_yamls}")

    if args.num_jobs > 1:
        assert args.job_idx < args.num_jobs, f"job_idx ({args.job_idx}) must be less than num_jobs ({args.num_jobs})"
        def chunk_list(data, n):
            avg = len(data) / float(n)
            chunks = []
            last = 0.0
            while last < len(data):
                chunks.append(data[int(last):int(last + avg)])
                last += avg
            return chunks

        train_yaml_chunks = chunk_list(all_train_yamls, args.num_jobs)
        train_yaml_chunks = sorted(train_yaml_chunks, key=lambda x: len(x), reverse=True)
        train_yaml_chunk = train_yaml_chunks[args.job_idx]
        print(f"Running job {args.job_idx}/{args.num_jobs} with {len(train_yaml_chunk)} training configs: {train_yaml_chunk}")
        all_train_yamls = train_yaml_chunk

    for train_yaml_path in all_train_yamls:
        args.train_yaml_path = train_yaml_path
        main(args)
