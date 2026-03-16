# The Delta Learning Hypothesis

**[The Delta Learning Hypothesis: Preference Tuning on Weak Data can Yield Strong Gains (COLM 2025)](https://arxiv.org/abs/2507.06187)**

Scott Geng, Hamish Ivison, Chun-Liang Li, Maarten Sap, Jerry Li, Ranjay Krishna, Pang Wei Koh

---

We show that paired data composed of individually weak responses can produce training gains exceeding what either response alone would yield. The key insight is that **the relative quality delta between two responses can drive learning** via preference algorithms that leverage paired gradients — even when SFT on either response individually would degrade performance.

As one practical demonstration, pairing responses from a 3B model (chosen) with responses from a 1.5B model (rejected) produces meaningful training signal for an 8B model, matching open-SOTA post-training performance across 11 benchmarks.

### 💡Olmo 3 post-training uses delta learning! Check out how we scaled delta learning to help build SOTA fully open reasoning and chat langauge models [in our tech report](https://arxiv.org/abs/2512.13961).

## Overview and Setup

This codebase is built on top of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). Any training codebase that supports length-normalized DPO will work (ex. [OpenInstruct](https://github.com/allenai/open-instruct)), so you may bring your own code if you wish. We intend this codebase to be used mainly for the sake of reference.

```bash
pip install -e .
```

We used an internal version of [Olmes](https://github.com/allenai/olmes) for all evaluations. Please check the corresponding work for eval settings (i.e. sampling parameters, prompts, etc.).

Training configs for reproducing our main experiments are provided under `configs/`. See below for details.

## Configs

Experiment configs are organized under `configs/dpo/` and `configs/sft/`:

| Directory | Description |
|---|---|
| `exp_posttrain/` | Main post-training experiments |
| `exp_analysis_qwen_ladder/` | Sweep over all chosen/rejected pairs across the Qwen 2.5 model ladder |
| `exp_num_sections/` | Controlled experiments varying the number of sections |
| `exp_self_improve/` | Controlled experiments on self-improvement |
| `exp_ultrafeedback_weak/` | UltraFeedback with weak model filtering |


## Data

Delta learning makes post-training data generation simple: take any prompt mix, pass it through models of different capability levels, and use the stronger model's responses as chosen and the weaker model's as rejected. We provide `scripts/generate_model_ladder_data.py` as a simple example.

All datasets used in our experiments are available on HuggingFace:

| Dataset | Used in |
|---|---|
| [`scottgeng00/delta_learning_model_ladder`](https://huggingface.co/datasets/scottgeng00/delta_learning_model_ladder) | `exp_posttrain/`, `exp_analysis_qwen_ladder/` |
| [`scottgeng00/delta_learning_num_sections`](https://huggingface.co/datasets/scottgeng00/delta_learning_num_sections) | `exp_num_sections/` |
| [`scottgeng00/delta_learning_num_sections_3section_tied`](https://huggingface.co/datasets/scottgeng00/delta_learning_num_sections_3section_tied) | `exp_num_sections/` |
| [`scottgeng00/delta_learning_ufweak`](https://huggingface.co/datasets/scottgeng00/delta_learning_ufweak) | `exp_ultrafeedback_weak/` |
| [`scottgeng00/delta_learning_tulu3-sft-mix_model_ladder`](https://huggingface.co/datasets/scottgeng00/delta_learning_tulu3-sft-mix_model_ladder) | `exp_self_improve/` |


## Training

We provide launch scripts for DPO and SFT training. Both scripts take a YAML config file and launch a DeepSpeed training job. By default, the repo root is used as the working directory.


**DPO:**
```bash
python scripts/launch_dpo_with_yaml.py \
--train_yaml_path configs/dpo/exp_posttrain/tulu3-8b-sft_dpo_chosen-qwen-2.5-3b-instruct_rejected-qwen-2.5-1.5b-instruct.yaml \
--num_gpus 8 --deepspeed_stage 2
```

**SFT:**
```bash
python scripts/launch_sft_with_yaml.py \
--train_yaml_path configs/sft/exp_analysis_qwen_ladder/tulu3-8b-sft_sft_qwen-2.5-3b-instruct.yaml \
--num_gpus 8 --deepspeed_stage 2
```

You can also pass a directory to run all configs in it sequentially:
```bash
python scripts/launch_dpo_with_yaml.py \
--train_yaml_path configs/dpo/exp_posttrain \
--num_gpus 8 --deepspeed_stage 2
```

**Key flags:**
- `--openrlhf_dir` — path to the OpenRLHF repo root (defaults to the repo root)
- `--deepspeed_stage` — DeepSpeed ZeRO stage (0–3)
- `--num_gpus` — number of GPUs
- `--train_overrides` — comma-separated key=value overrides for the YAML config (e.g. `learning_rate=1e-7,beta=5`)
- `--num_jobs` / `--job_idx` — split configs across SLURM array jobs

## Citation
If you find our work useful, please cite:

```bibtex
@article{geng2025delta,
  title={The Delta Learning Hypothesis: Preference Tuning on Weak Data can Yield Strong Gains},
  author={Geng, Scott and Ivison, Hamish and Li, Chun-Liang Li and Sap, Maarten and Li, Jerry and Krishna, Ranjay and Koh, Pang Wei},
  journal={arXiv preprint arXiv:2507.06187},
  year={2025}
}
```
