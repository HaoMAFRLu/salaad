# SALAAD

SALAAD is research code for training language models with sparse plus low-rank
weight decompositions. The current implementation focuses on LLaMA-style causal
language models and applies an ADMM-style training objective to selected weight
matrices:

```text
X ~= L + S
```

where `X` is the trainable weight, `L` is a low-rank component, and `S` is a
sparse component.

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

Start with the debug configuration:

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

## Evaluation

The repository includes scripts for reconstructing selected layers from saved
`L` and `S` matrices and for running LM Evaluation Harness tasks after resaving a
checkpoint in Hugging Face format.

Some evaluation scripts are still experiment-oriented and may need path or task
edits for new runs.

## Notes

This is research code. Start with `llama_debug`, verify dataset streaming in
your environment, and avoid committing generated checkpoints or experiment
outputs.
