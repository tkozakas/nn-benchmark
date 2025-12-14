#!/usr/bin/env bash
set -e

# Configuration
HPC_USER="toko7940"
HPC_HOST="hpc.mif.vu.lt"
SSH_KEY="~/.ssh/id_ed25519"
REMOTE_DIR="/scratch/lustre/home/${HPC_USER}/nn-benchmark"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse arguments
EXPERIMENT="${1:-full}"

SSH_CMD="ssh -i $SSH_KEY ${HPC_USER}@${HPC_HOST}"
SCP_CMD="scp -i $SSH_KEY"

echo "=== HPC Runner ==="
echo "Experiment: $EXPERIMENT"
echo "Options: learning-rate | optimizer | scheduler | batch-size | architecture | hpo | architecture-hpo | full"

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
JOBID=$($SSH_CMD "cd $REMOTE_DIR && mkdir -p logs && sbatch --parsable run_experiment.sh $EXPERIMENT")
echo "Job submitted with ID: $JOBID"

LOGFILE="logs/tinyimagenet_benchmark_${JOBID}.out"

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
echo "Log file: $LOCAL_DIR/logs/tinyimagenet_benchmark_${JOBID}.out"
