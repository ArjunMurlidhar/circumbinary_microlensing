#!/bin/bash

# Helper script to submit all SLURM jobs for a given run
# Usage: ./submit_all_jobs.sh <run_name>

# Fixed paths
SCRATCH_BASE="/fs/scratch/PAS3230"
FINAL_OUTPUT_DIR="/users/PAS3230/arjunm/circumbinary_microlensing/oom_detectability"

if [ -z "$1" ]; then
    echo "Usage: $0 <run_name>"
    echo ""
    echo "This script submits all SLURM jobs for the specified run."
    echo "Make sure you've run split_and_submit.sh first!"
    exit 1
fi

RUN_NAME="$1"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
SLURM_DIR="$PROJECT_DIR/slurm_scripts"

# Scratch log directory
SCRATCH_LOG_DIR="$SCRATCH_BASE/$RUN_NAME/logs"

# Check if SLURM directory exists
if [ ! -d "$SLURM_DIR" ]; then
    echo "Error: SLURM directory not found: $SLURM_DIR"
    echo "Have you run split_and_submit.sh yet?"
    exit 1
fi

# Find all SLURM scripts for this run
SLURM_SCRIPTS=("$SLURM_DIR/${RUN_NAME}_job"*.slurm)

if [ ! -f "${SLURM_SCRIPTS[0]}" ]; then
    echo "Error: No SLURM scripts found for run: $RUN_NAME"
    echo "Looking in: $SLURM_DIR"
    exit 1
fi

echo "=========================================="
echo "Submitting SLURM jobs for: $RUN_NAME"
echo "=========================================="
echo ""

JOB_IDS=()
SKIPPED_JOBS=()
for script in "${SLURM_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        # Extract job name from script filename (e.g., myrun_job0.slurm -> myrun_job0)
        JOB_NAME=$(basename "$script" .slurm)
        
        # Check if results file already exists in final output directory
        RESULTS_FILE="$FINAL_OUTPUT_DIR/${JOB_NAME}_results.csv"
        if [ -f "$RESULTS_FILE" ]; then
            echo "Skipping: $(basename "$script") - results already exist"
            echo "  Found: $RESULTS_FILE"
            SKIPPED_JOBS+=("$JOB_NAME")
            continue
        fi
        
        echo "Submitting: $(basename "$script")"
        JOB_OUTPUT=$(sbatch "$script")
        JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
        JOB_IDS+=("$JOB_ID")
        echo "  Job ID: $JOB_ID"
    fi
done

echo ""
echo "=========================================="
echo "Submitted ${#JOB_IDS[@]} jobs"
if [ ${#SKIPPED_JOBS[@]} -gt 0 ]; then
    echo "Skipped ${#SKIPPED_JOBS[@]} jobs (results already exist)"
fi
echo "=========================================="
echo ""

if [ ${#JOB_IDS[@]} -eq 0 ]; then
    echo "No jobs were submitted - all results already exist!"
    exit 0
fi

echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo ""
echo "Cancel all jobs:"
echo "  scancel $(IFS=' '; echo "${JOB_IDS[*]}")"
echo ""
echo "Check logs in scratch:"
echo "  $SCRATCH_LOG_DIR/${RUN_NAME}_job*.log"
echo "  $SCRATCH_LOG_DIR/${RUN_NAME}_job*.err"
echo ""
echo "Check job status:"
echo "  ./check_job_status.sh $RUN_NAME"
echo "=========================================="
