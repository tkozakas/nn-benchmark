#!/usr/bin/env bash
set -e

# Configuration
HPC_USER="toko7940"
HPC_HOST="hpc.mif.vu.lt"
SSH_KEY="~/.ssh/id_ed25519"
REMOTE_DIR="/scratch/lustre/home/${HPC_USER}/nn-benchmark"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Default values
ARCHITECTURE="DenseNet121"
EPOCHS="20"
PATIENCE="5"
N_TRIALS="30"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --architecture) ARCHITECTURE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --n-trials) N_TRIALS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: ./hpc_optimize.sh [OPTIONS]"
      echo ""
      echo "Runs Optuna hyperparameter optimization to find the best configuration."
      echo ""
      echo "Optimizes:"
      echo "  - Pretrained vs from scratch"
      echo "  - Augmentation: none, mixup, cutmix"
      echo "  - Learning rate (1e-5 to 1e-2)"
      echo "  - Batch size (64, 128, 256)"
      echo "  - Weight decay (1e-6 to 1e-2)"
      echo "  - Optimizer (adam, adamw, sgd)"
      echo ""
      echo "Options:"
      echo "  --architecture   Model architecture (default: DenseNet121)"
      echo "  --epochs         Epochs per trial (default: 20)"
      echo "  --patience       Early stopping patience (default: 5)"
      echo "  --n-trials       Number of Optuna trials (default: 30)"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

SSH_CMD="ssh -i $SSH_KEY ${HPC_USER}@${HPC_HOST}"
SCP_CMD="scp -i $SSH_KEY"

echo "=== HPC Optuna Optimizer ==="
echo "Architecture: $ARCHITECTURE"
echo "Epochs/trial: $EPOCHS"
echo "Patience:     $PATIENCE"
echo "N trials:     $N_TRIALS"
echo ""
echo "Will optimize:"
echo "  - pretrained: [True, False]"
echo "  - augmentation: [none, mixup, cutmix]"
echo "  - lr: [1e-5, 1e-2]"
echo "  - batch_size: [64, 128, 256]"
echo "  - weight_decay: [1e-6, 1e-2]"
echo "  - optimizer: [adam, adamw, sgd]"

# Step 1: Sync local code to HPC
echo ""
echo "=== Syncing code to HPC ==="
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'test_data' --exclude 'trained' --exclude 'logs/*.out' \
  -e "ssh -i $SSH_KEY" \
  "$LOCAL_DIR/" "${HPC_USER}@${HPC_HOST}:${REMOTE_DIR}/"

# Step 2: Submit job and get job ID
echo ""
echo "=== Submitting job to HPC ==="
JOBID=$($SSH_CMD "cd $REMOTE_DIR && mkdir -p logs && sbatch --parsable run_optimize.sh \
  $ARCHITECTURE $EPOCHS $PATIENCE $N_TRIALS")
echo "Job submitted with ID: $JOBID"

LOGFILE="logs/optimization_benchmark_${JOBID}.out"

# Step 3: Wait for log file to appear and stream it
echo ""
echo "=== Waiting for job to start ==="
while ! $SSH_CMD "test -f $REMOTE_DIR/$LOGFILE" 2>/dev/null; do
  sleep 2
  JOB_STATE=$($SSH_CMD "squeue -j $JOBID -h -o %T 2>/dev/null || echo 'UNKNOWN'")
  echo "Job state: $JOB_STATE"
  if [[ "$JOB_STATE" == "UNKNOWN" || -z "$JOB_STATE" ]]; then
    echo "Job may have completed or failed before log was created"
    break
  fi
done

echo ""
echo "=== Streaming logs (Ctrl+C to stop streaming, job continues) ==="
$SSH_CMD "tail -f $REMOTE_DIR/$LOGFILE" &
TAIL_PID=$!

# Monitor job status
while true; do
  sleep 10
  JOB_STATE=$($SSH_CMD "squeue -j $JOBID -h -o %T 2>/dev/null || echo ''")
  if [[ -z "$JOB_STATE" ]]; then
    echo ""
    echo "=== Job $JOBID completed ==="
    kill $TAIL_PID 2>/dev/null || true
    break
  fi
done

# Step 4: Copy results back
echo ""
echo "=== Copying results back ==="

mkdir -p "$LOCAL_DIR/logs"
$SCP_CMD "${HPC_USER}@${HPC_HOST}:${REMOTE_DIR}/${LOGFILE}" "$LOCAL_DIR/logs/" 2>/dev/null || true

echo "Copying test_data..."
$SCP_CMD -r "${HPC_USER}@${HPC_HOST}:${REMOTE_DIR}/test_data" "$LOCAL_DIR/" 2>/dev/null || echo "No test_data to copy"

echo ""
echo "=== Done! ==="
echo "Log file: $LOCAL_DIR/logs/optimization_benchmark_${JOBID}.out"
echo "Results: $LOCAL_DIR/test_data/optuna_${ARCHITECTURE,,}.csv"
echo "Best config: $LOCAL_DIR/test_data/optuna_${ARCHITECTURE,,}_best.csv"
