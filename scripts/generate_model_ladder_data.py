"""
Generate model ladder data by passing prompts through multiple models.
Each model is loaded in a subprocess so GPU memory is freed between runs.
Usage: python scripts/generate_model_ladder_data.py --input_dataset <hf_dataset> --output_dataset <hf_dataset>
"""
import argparse
import json
import subprocess
import sys
import tempfile
import os
from datasets import load_dataset, Dataset

MODEL_MAP = {
    "qwen-2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-2.5-3b-instruct":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen-2.5-7b-instruct":   "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-14b-instruct":  "Qwen/Qwen2.5-14B-Instruct",
    "qwen-2.5-32b-instruct":  "Qwen/Qwen2.5-32B-Instruct",
    "qwen-2.5-72b-instruct":  "Qwen/Qwen2.5-72B-Instruct",
}

WORKER_CODE = """
import sys, json
from vllm import LLM, SamplingParams

model_name, prompts_file, output_file, max_tokens, tp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
prompts = json.load(open(prompts_file))
llm = LLM(model=model_name, tensor_parallel_size=tp)
tokenizer = llm.get_tokenizer()
chats = [[{"role": "user", "content": p}] for p in prompts]
formatted = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
outputs = llm.generate(formatted, SamplingParams(max_tokens=max_tokens, temperature=0.0))
json.dump([o.outputs[0].text for o in outputs], open(output_file, "w"))
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, required=True)
    parser.add_argument("--output_dataset", type=str, required=True)
    parser.add_argument("--prompts_column", type=str, default="prompt")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = load_dataset(args.input_dataset, split="train")
    prompts = dataset[args.prompts_column]
    data = {args.prompts_column: prompts}

    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_file = os.path.join(tmpdir, "prompts.json")
        json.dump(prompts, open(prompts_file, "w"))

        for short_name, hf_name in MODEL_MAP.items():
            print(f"Generating with {short_name}...")
            output_file = os.path.join(tmpdir, f"{short_name}.json")
            subprocess.run(
                [sys.executable, "-c", WORKER_CODE, hf_name, prompts_file, output_file,
                 str(args.max_tokens), str(args.tensor_parallel_size)],
                check=True
            )
            data[short_name] = json.load(open(output_file))

    Dataset.from_dict(data).push_to_hub(args.output_dataset)
    print(f"Pushed to {args.output_dataset}")

if __name__ == "__main__":
    main()
