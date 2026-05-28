# SALAAD

This repository contains the official implementation of **SALAAD: Sparse And
Low-Rank Adaptation via ADMM for Large Language Model Inference**. SALAAD is a
plug-and-play framework for inducing sparse plus low-rank (SLR) structure during
language-model pretraining, with the goal of supporting elastic deployment under
different memory and compute budgets.

SALAAD operates in weight space and does not require architectural changes to the
underlying Transformer model. During training, selected weight matrices are
coupled to structured surrogate variables through an ADMM-style augmented
Lagrangian objective:

```text
X ~= L + S
```

where `X` is the trainable weight, `L` is a low-rank component, and `S` is a
sparse component. A single SALAAD-trained checkpoint can then be adapted to a
continuous range of parameter budgets by adjusting the effective rank and
sparsity of the learned surrogate weights, without retraining or modifying the
model architecture.

The code currently focuses on LLaMA-style causal language models and includes
training, evaluation, Hugging Face export, and LM Evaluation Harness workflows.

## Repository Layout

```text
salaad/       Core trainer, ADMM solver, operators, and utilities
models/       Local LLaMA model implementation
dataloaders/  Iterable C4/tokenization utilities
configs/      Small example training and model configs
scripts/      Training, evaluation, resave, and lm-eval entry points
tests/        Lightweight smoke tests
```

The main training path is:

```text
scripts/train_salad.py
  -> salaad/register.py
  -> salaad/trainer_salad.py
  -> salaad/salad_solver.py
```

## Installation

Use Python 3.9+ and install a PyTorch build that matches your CUDA environment
before installing this package. See the PyTorch installation instructions for
the correct command for your system.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install PyTorch separately for your CUDA/CPU environment first.
pip install -e ".[dev]"
```

Training streams C4 from Hugging Face. If you need gated resources or hit rate
limits, authenticate before running:

```bash
huggingface-cli login
```

Weights & Biases logging is optional. If enabled in a config, set:

```bash
export WANDB_API_KEY=...
export WANDB_ENTITY=...
```

## Quickstart

Run the lightweight solver smoke test first:

```bash
python -m pytest tests/test_smoke.py -q
```

For real training, start with the debug configuration. This command streams C4
from Hugging Face and expects a working PyTorch distributed/CUDA setup:


```bash
torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
  scripts/train_salad.py \
  --cfg_version llama_debug \
  --folder debug
```

Outputs are written to:

```text
data/<folder>/<cfg_version>/<timestamp>/
```

Typical outputs include `model.pth`, copied config files, `layer_info.pkl`, and
rank-local `matrix_rank<N>.pkl` files containing the low-rank, sparse, and dual
variables.

## Configuration

Training configs live in `configs/*.yaml`; matching architecture configs live in
`configs/*_model.json`.

Important fields:

- `training_mode`: `salad` for low-rank/sparse training or `vanilla` for normal training.
- `num_total_iters`: total optimizer updates.
- `num_freq`: how often to run ADMM updates.
- `layers`: model layers to decompose.
- `rate_rank`: target low-rank ratio for a layer.
- `rate_sparsity`: target sparse density for a layer.
- `rho_dict`, `alpha_dict`, `beta_dict`: ADMM penalty and adaptive threshold settings.

## Evaluation Workflow

Evaluate a trained SALAAD run directly from its output directory:

```bash
python scripts/evaluation.py \
  --run_dir data/debug/llama_debug/<timestamp> \
  --target_params 6.5
```

Resave the checkpoint in Hugging Face format:

```bash
python scripts/resave_model.py \
  --run_dir data/debug/llama_debug/<timestamp> \
  --target_params 6.5 \
  --gamma 0.5
```

This writes:

```text
data/debug/llama_debug/<timestamp>/model_resave/vanilla/
data/debug/llama_debug/<timestamp>/model_resave/surrogate/
```

Run LM Evaluation Harness on a resaved model:

```bash
python scripts/run_lm_eval.py \
  --model_dir data/debug/llama_debug/<timestamp>/model_resave \
  --variant both \
  --tasks piqa boolq \
  --batch_size 8
```

Use `--variant direct` if `--model_dir` points directly to a Hugging Face model
folder rather than a directory containing `vanilla/` and `surrogate/`.

## Notes

This is research code. Start with `llama_debug`, verify dataset streaming in
your environment, and avoid committing generated checkpoints or experiment
outputs.

## Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{ma2026salaad,
  title     = {{SALAAD}: Sparse And Low-Rank Adaptation via {ADMM} for Large Language Model Inference},
  author    = {Ma, Hao and Bal, Melis Ilayda and Zhang, Liang and Li, Bingcong and He, Niao and Zeilinger, Melanie and Muehlebach, Michael},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  address   = {Seoul, South Korea},
  publisher = {PMLR}
}
```
