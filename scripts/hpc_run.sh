#!/usr/bin/env bash
set -e

# Configuration
HPC_HOST="hpc.mif.vu.lt"
SSH_KEY="~/.ssh/id_ed25519"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
HPC_USER=""
ARCHITECTURE="ResNet50"
K_FOLDS="3"
EPOCHS="20"
BATCH_SIZE="2048"
LR="0.001"
PATIENCE="5"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --user) HPC_USER="$2"; shift 2 ;;
    --architecture) ARCHITECTURE="$2"; shift 2 ;;
    --k-folds) K_FOLDS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: ./hpc_run.sh --user USERNAME [OPTIONS]"
      echo ""
      echo "Required:"
      echo "  --user           HPC username (required)"
      echo ""
      echo "Options:"
      echo "  --architecture   Initial architecture (default: ResNet50)"
      echo "  --k-folds        Number of folds (default: 3)"
      echo "  --epochs         Number of epochs (default: 20)"
      echo "  --batch-size     Batch size (default: 2048)"
      echo "  --lr             Initial learning rate (default: 0.001)"
      echo "  --patience       Early stopping patience (default: 5)"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$HPC_USER" ]]; then
  echo "Error: --user is required"
  echo "Usage: ./hpc_run.sh --user USERNAME [OPTIONS]"
  exit 1
fi

REMOTE_DIR="/scratch/lustre/home/${HPC_USER}/nn-benchmark"

SSH_CMD="ssh -i $SSH_KEY ${HPC_USER}@${HPC_HOST}"
SCP_CMD="scp -i $SSH_KEY"

echo "=== HPC Runner ==="
echo "User:         $HPC_USER"
echo "Architecture: $ARCHITECTURE"
echo "K-Folds:      $K_FOLDS"
echo "Epochs:       $EPOCHS"
echo "Batch Size:   $BATCH_SIZE"
echo "LR:           $LR"
echo "Patience:     $PATIENCE"

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
JOBID=$($SSH_CMD "cd $REMOTE_DIR && mkdir -p logs && sbatch --parsable scripts/run_experiment.sh \
  $ARCHITECTURE $K_FOLDS $EPOCHS $BATCH_SIZE $LR $PATIENCE")
echo "Job submitted with ID: $JOBID"

LOGFILE="logs/experiment_${JOBID}.out"

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

echo "Copying trained models..."
$SCP_CMD -r "${HPC_USER}@${HPC_HOST}:${REMOTE_DIR}/trained" "$LOCAL_DIR/" 2>/dev/null || echo "No trained models to copy"

echo ""
echo "=== Done! ==="
echo "Log file: $LOCAL_DIR/logs/experiment_${JOBID}.out"
