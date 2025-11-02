#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=tinyimagenet_benchmark
#SBATCH --output=logs/tinyimagenet_benchmark_%j.out

# load UV Python & your venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

rm -rf .venv
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install torch torchvision torchaudio

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=16

python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

cd src
python experiment.py \
  --architecture ResNet50 \
  --device cuda \
  --cpu-workers 16 \
  --k-folds 3 \
  --epochs 20 \
  --batch-size 2048 \
  --lr 0.001 \
  --patience 5 \
  --subsample-size None
