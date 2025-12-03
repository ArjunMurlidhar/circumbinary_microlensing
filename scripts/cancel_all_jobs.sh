#!/bin/bash

# Helper script to cancel all SLURM jobs for a given run
# Usage: ./cancel_all_jobs.sh <run_name>

if [ -z "$1" ]; then
    echo "Usage: $0 <run_name>"
    echo ""
    echo "This script cancels all SLURM jobs for the specified run."
    exit 1
fi

RUN_NAME="$1"

echo "=========================================="
echo "Cancelling jobs for: $RUN_NAME"
echo "=========================================="
echo ""

# Get all job IDs for this run
JOB_IDS=$(squeue -u $USER -o "%.18i %.12j" | grep "$RUN_NAME" | awk '{print $1}')

if [ -z "$JOB_IDS" ]; then
    echo "No active jobs found for run: $RUN_NAME"
    exit 0
fi

# Count jobs
NUM_JOBS=$(echo "$JOB_IDS" | wc -l | tr -d ' ')

echo "Found $NUM_JOBS job(s) to cancel:"
echo ""

# Show jobs that will be cancelled
squeue -u $USER -o "%.18i %.12j %.8T %.10M %.10l %.6D %R" | grep "$RUN_NAME"

echo ""
read -p "Are you sure you want to cancel these jobs? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Cancelling jobs..."
    for job_id in $JOB_IDS; do
        echo "  Cancelling job $job_id"
        scancel $job_id
    done
    echo ""
    echo "All jobs cancelled!"
else
    echo ""
    echo "Cancelled. No jobs were terminated."
fi

echo "=========================================="

