#!/bin/bash

# Helper script to check status of all jobs for a given run
# Usage: ./check_job_status.sh <run_name>

if [ -z "$1" ]; then
    echo "Usage: $0 <run_name>"
    echo ""
    echo "This script checks the status of all SLURM jobs for the specified run."
    exit 1
fi

RUN_NAME="$1"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
SLURM_DIR="$PROJECT_DIR/slurm_scripts"

echo "=========================================="
echo "Job Status for: $RUN_NAME"
echo "=========================================="
echo ""

# Check queue status
echo "Jobs in queue:"
squeue -u $USER -o "%.18i %.12j %.8T %.10M %.10l %.6D %R" | grep "$RUN_NAME" || echo "  No jobs found in queue"

echo ""
echo "----------------------------------------"
echo "Log file summary:"
echo "----------------------------------------"
echo ""

# Check log files
LOG_FILES=("$SLURM_DIR/${RUN_NAME}_job"*.log)
ERR_FILES=("$SLURM_DIR/${RUN_NAME}_job"*.err)

if [ -f "${LOG_FILES[0]}" ]; then
    for log in "${LOG_FILES[@]}"; do
        if [ -f "$log" ]; then
            job_name=$(basename "$log" .log)
            err_file="${log%.log}.err"
            
            # Check if job completed
            if grep -q "Job completed" "$log" 2>/dev/null; then
                exit_code=$(grep "Exit code:" "$log" | tail -1 | awk '{print $NF}')
                if [ "$exit_code" = "0" ]; then
                    status="✓ COMPLETED"
                else
                    status="✗ FAILED (exit $exit_code)"
                fi
            elif [ -s "$log" ]; then
                status="⟳ RUNNING"
            else
                status="⋯ PENDING/NO OUTPUT"
            fi
            
            # Check error file
            err_size=0
            if [ -f "$err_file" ] && [ -s "$err_file" ]; then
                err_size=$(wc -l < "$err_file" | tr -d ' ')
            fi
            
            printf "%-30s %s" "$job_name:" "$status"
            if [ $err_size -gt 0 ]; then
                printf " (errors: %d lines)" $err_size
            fi
            echo ""
        fi
    done
else
    echo "No log files found for run: $RUN_NAME"
    echo "Have you submitted the jobs yet?"
fi

echo ""
echo "=========================================="
echo "Recent errors (last 5 lines from each .err file):"
echo "=========================================="
echo ""

for err in "${ERR_FILES[@]}"; do
    if [ -f "$err" ] && [ -s "$err" ]; then
        echo "--- $(basename "$err") ---"
        tail -5 "$err"
        echo ""
    fi
done

if [ ! -f "${ERR_FILES[0]}" ] || [ ! -s "${ERR_FILES[0]}" ]; then
    echo "No error files with content found."
fi

echo ""
echo "=========================================="
echo "Detailed logs available at:"
echo "  $SLURM_DIR/${RUN_NAME}_job*.log"
echo "  $SLURM_DIR/${RUN_NAME}_job*.err"
echo "=========================================="

