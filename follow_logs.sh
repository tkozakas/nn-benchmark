#!/usr/bin/env bash
mkdir -p logs

# follow_logs.sh
if [[ -z "$1" ]]; then
  echo "Usage: $0 <jobid>"
  exit 1
fi
JOBID=$1
LOGFILE=logs/ocr_benchmark_${JOBID}.out

# wait until the file appears (in case Slurm hasn’t created it yet)
while [[ ! -f "$LOGFILE" ]]; do
  sleep 1
done

# stream it
echo "=== following $LOGFILE ==="
tail -f "$LOGFILE"
