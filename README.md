# CNN Architecture Benchmark (TinyImageNet)

## Project Goals

Benchmark multiple off‑the‑shelf torchvision CNN architectures (see `config.py`) on the TinyImageNet dataset, comparing
classification performance, efficiency (latency, throughput, resource usage) and core training hyperparameters.
The pipeline automates comparative experiments for:

1. Learning rate sweep
2. Optimizer selection
3. LR scheduler selection
4. Weight decay (L2) regularisation strength
5. Batch size impact
6. Final architecture comparison
7. Optuna hyperparameter optimisation (LR, WD, batch size, optimizer, scheduler)

## Environment

- GPU backend: Works with CUDA or ROCm (tested previously on ROCm 6.x)
- Python: 3.12

If using an AMD GPU that requires an override (example shown for some RDNA2 cards):
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

## Installation
### Install [uv](https://github.com/astral-sh/uv)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```
### Install dependencies
```bash
uv pip install -r requirements.txt
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
# (Optional) explicit torch install if you need a specific CUDA wheel
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Quick Experiment Run
Example: run the full comparison pipeline starting from ResNet18.
```bash
cd src
python experiment.py \
  --architecture ResNet18 \
  --device cuda \
  --cpu-workers 8 \
  --k-folds 3 \
  --epochs 10 \
  --batch-size 128 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --patience 5
```
Supported architectures are declared in `config.py` (default: GoogleNet, ResNet18, ResNet50, DenseNet121). Extend that
mapping to add more.

## Individual Training (debug / smaller scope)
```bash
cd src
python train.py --architecture ResNet18 --device cuda --epochs 5 --batch-size 128
```

## HPC Batch Script Example
```bash
chmod +x *.sh
./setup.sh
./follow_logs.sh $(sbatch --parsable run_experiment.sh)
```

## Retrieving Results
CSV summaries & plots are written under `test_data/` and trained model weights under `trained/`.