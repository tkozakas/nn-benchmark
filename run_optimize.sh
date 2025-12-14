#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=optimization_benchmark
#SBATCH --output=logs/optimization_benchmark_%j.out

# Usage: ./run_optimize.sh [architecture] [k-folds] [epochs] [batch-size] [lr] [patience]
ARCHITECTURE=${1:-DenseNet121}
K_FOLDS=${2:-3}
EPOCHS=${3:-20}
BATCH_SIZE=${4:-128}
LR=${5:-0.001}
PATIENCE=${6:-5}

if ! command -v uv &> /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env
fi

if [[ ! -d .venv ]] || [[ requirements.txt -nt .venv ]]; then
  echo "Setting up virtual environment..."
  uv python install 3.12
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -r requirements.txt
  uv pip install torch torchvision torchaudio
  touch .venv
else
  echo "Using cached virtual environment..."
  source .venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=16

python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

cd src
python optimize.py \
  --architecture "$ARCHITECTURE" \
  --device cuda \
  --cpu-workers 16 \
  --k-folds "$K_FOLDS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --patience "$PATIENCE"
