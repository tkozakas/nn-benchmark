#!/bin/bash
#SBATCH --job-name=dataset_setup
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

DATASET_DIR="dataset"
ZIP_NAME="tiny-imagenet.zip"
DATASET_URL="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
EXTRACTED_DIR="tiny-imagenet-200"

mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"
curl -L -o "$ZIP_NAME" "$DATASET_URL"
unzip -q "$ZIP_NAME"

rm -f "$ZIP_NAME"
