#!/bin/bash
#SBATCH --job-name=dataset_setup
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

# Create logs directory
mkdir -p logs

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node: $SLURM_NODELIST"
echo "[SLURM] Starting at: $(date)"

DATASET_DIR="dataset"
ZIP_NAME="tiny-imagenet.zip"
DATASET_URL="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
EXTRACTED_DIR="tiny-imagenet-200"

if [ -d "$DATASET_DIR/$EXTRACTED_DIR/train" ]; then
  echo "[INFO] Dataset already present at $DATASET_DIR/$EXTRACTED_DIR. Skipping download."
  exit 0
fi

mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "[INFO] Downloading TinyImageNet from Stanford mirror..."
curl -L -o "$ZIP_NAME" "$DATASET_URL"

echo "[INFO] Extracting only train directory (using ${SLURM_CPUS_PER_TASK:-4} CPU cores)..."
# Extract ONLY the train folder to save time
unzip -q "$ZIP_NAME" "tiny-imagenet-200/train/*" 2>/dev/null || unzip -q "$ZIP_NAME" "*/train/*"

if [ ! -d "$EXTRACTED_DIR" ]; then
  MOVED_DIR=$(find . -maxdepth 2 -type d -name "$EXTRACTED_DIR" | head -n 1 || true)
  if [ -n "$MOVED_DIR" ] && [ "$MOVED_DIR" != "./$EXTRACTED_DIR" ]; then
    mv "$MOVED_DIR" "$EXTRACTED_DIR"
  fi
fi

if [ -d "$EXTRACTED_DIR/train" ]; then
  echo "[INFO] Dataset ready: $DATASET_DIR/$EXTRACTED_DIR/train"
  echo "[INFO] Note: Only train data extracted. Val/test not needed for current pipeline."
else
  echo "[ERROR] Expected $EXTRACTED_DIR/train not found after unzip." >&2
  exit 1
fi

rm -f "$ZIP_NAME"
echo "[SLURM] Setup complete at: $(date)"
