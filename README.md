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
  --patience 5
```
Supported architectures are declared in `config.py` (default: GoogleNet, ResNet18, ResNet50, DenseNet121). Extend that
mapping to add more.

## Run on VU HPC

### Setup SSH Key
1. Generate an SSH key (if you haven't already):
   ```bash
   ssh-keygen -t ed25519
   chmod 600 ~/.ssh/id_ed25519
   ```
2. Upload your public key (`~/.ssh/id_ed25519.pub`) to [MIF LDAP](https://ldap.mif.vu.lt)

3. Set your username and test connection:
   ```bash
   HPC_USER=YOUR_VU_COMPUTER_USERNAME
   ssh -i ~/.ssh/id_ed25519 $HPC_USER@hpc.mif.vu.lt
   ```

### Run Experiment
Run experiments from your local machine (syncs code, submits job, streams logs, copies results back):
```bash
./hpc_run.sh [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--architecture` | `ResNet50` | Initial architecture |
| `--k-folds` | `3` | Number of CV folds |
| `--epochs` | `20` | Number of epochs |
| `--batch-size` | `2048` | Batch size |
| `--lr` | `0.001` | Initial learning rate |
| `--patience` | `5` | Early stopping patience |

**Example:**
```bash
./hpc_run.sh --architecture ResNet18 --epochs 10 --batch-size 512
```

### Run Optimization Experiment
Compare pretrained vs from-scratch training with data augmentation (Mixup/CutMix):
```bash
./hpc_optimize.sh [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--architecture` | `DenseNet121` | Model architecture |
| `--k-folds` | `3` | Number of CV folds |
| `--epochs` | `20` | Number of epochs |
| `--batch-size` | `128` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--patience` | `5` | Early stopping patience |

**Compares 6 configurations:**
1. From scratch (baseline)
2. From scratch + Mixup
3. From scratch + CutMix
4. Pretrained (transfer learning)
5. Pretrained + Mixup
6. Pretrained + CutMix

**Example:**
```bash
./hpc_optimize.sh --architecture DenseNet121 --epochs 15 --batch-size 256
```

### Cancel Jobs
```bash
ssh -i ~/.ssh/id_ed25519 $HPC_USER@hpc.mif.vu.lt "scancel -u $HPC_USER"
```

