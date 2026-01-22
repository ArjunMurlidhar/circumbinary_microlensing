#!/bin/bash

# Script to split input file and create SLURM jobs for run_sep_det.py
# Uses scratch storage for I/O during job execution, then moves to final destination
# Usage: ./split_and_submit.sh --input_file <file> --num_jobs <n> --run_name <name> [--num_cores <n>] [--mag_plot]

# Fixed paths
SCRATCH_BASE="/fs/scratch/PAS3230"
FINAL_OUTPUT_DIR="/users/PAS3230/arjunm/circumbinary_microlensing/oom_detectability"

# Default values
NUM_CORES=1
MAG_PLOT_FLAG=""
SLURM_TIME="24:00:00"
SLURM_MEM="16G"
SLURM_PARTITION="normal"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --num_jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --num_cores)
            NUM_CORES="$2"
            shift 2
            ;;
        --mag_plot)
            MAG_PLOT_FLAG="--mag_plot"
            shift
            ;;
        --slurm_time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --slurm_mem)
            SLURM_MEM="$2"
            shift 2
            ;;
        --slurm_partition)
            SLURM_PARTITION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$INPUT_FILE" ] || [ -z "$NUM_JOBS" ] || [ -z "$RUN_NAME" ]; then
    echo "Usage: $0 --input_file <file> --num_jobs <n> --run_name <name> [--num_cores <n>] [--mag_plot] [--slurm_time <time>] [--slurm_mem <mem>] [--slurm_partition <partition>]"
    echo ""
    echo "Required arguments:"
    echo "  --input_file        Path to input CSV file"
    echo "  --num_jobs          Number of jobs to split into"
    echo "  --run_name          Global run name"
    echo ""
    echo "Optional arguments:"
    echo "  --num_cores         Number of cores per job (default: 1)"
    echo "  --mag_plot          Flag to plot magnification maps"
    echo "  --slurm_time        SLURM time limit (default: 24:00:00)"
    echo "  --slurm_mem         SLURM memory per job (default: 16G)"
    echo "  --slurm_partition   SLURM partition (default: normal)"
    echo ""
    echo "Fixed paths:"
    echo "  Scratch base:       $SCRATCH_BASE"
    echo "  Final output:       $FINAL_OUTPUT_DIR"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
SLURM_DIR="$PROJECT_DIR/slurm_scripts"

# Scratch directory structure
SCRATCH_RUN_DIR="$SCRATCH_BASE/$RUN_NAME"
SCRATCH_OUTPUT_DIR="$SCRATCH_RUN_DIR/outputs"
SCRATCH_LOG_DIR="$SCRATCH_RUN_DIR/logs"
SCRATCH_CODE_DIR="$SCRATCH_RUN_DIR/code"
SCRATCH_INPUT_DIR="$SCRATCH_RUN_DIR/split_inputs"

# Create necessary directories
mkdir -p "$FINAL_OUTPUT_DIR"
mkdir -p "$SLURM_DIR"
mkdir -p "$SCRATCH_OUTPUT_DIR"
mkdir -p "$SCRATCH_LOG_DIR"
mkdir -p "$SCRATCH_CODE_DIR"
mkdir -p "$SCRATCH_INPUT_DIR"

# Copy code directory to scratch
echo "Copying code to scratch..."
cp -r "$PROJECT_DIR/code/"* "$SCRATCH_CODE_DIR/"

echo "=========================================="
echo "Split and Submit Configuration"
echo "=========================================="
echo "Input file:        $INPUT_FILE"
echo "Number of jobs:    $NUM_JOBS"
echo "Run name:          $RUN_NAME"
echo "Cores per job:     $NUM_CORES"
echo "Mag plot:          ${MAG_PLOT_FLAG:-No}"
echo "SLURM time:        $SLURM_TIME"
echo "SLURM memory:      $SLURM_MEM"
echo "SLURM partition:   $SLURM_PARTITION"
echo ""
echo "Scratch directories:"
echo "  Base:            $SCRATCH_RUN_DIR"
echo "  Outputs:         $SCRATCH_OUTPUT_DIR"
echo "  Logs:            $SCRATCH_LOG_DIR"
echo "  Code:            $SCRATCH_CODE_DIR"
echo "  Inputs:          $SCRATCH_INPUT_DIR"
echo ""
echo "Final output:      $FINAL_OUTPUT_DIR"
echo "=========================================="
echo ""

# Count total lines (excluding header)
HEADER=$(head -n 1 "$INPUT_FILE")
TOTAL_LINES=$(tail -n +2 "$INPUT_FILE" | wc -l | tr -d ' ')

echo "Total data lines in input file: $TOTAL_LINES"

if [ "$TOTAL_LINES" -lt "$NUM_JOBS" ]; then
    echo "Warning: Number of jobs ($NUM_JOBS) is greater than data lines ($TOTAL_LINES)."
    echo "Adjusting number of jobs to $TOTAL_LINES"
    NUM_JOBS=$TOTAL_LINES
fi

# Calculate lines per job
LINES_PER_JOB=$((TOTAL_LINES / NUM_JOBS))
REMAINDER=$((TOTAL_LINES % NUM_JOBS))

echo "Lines per job: ~$LINES_PER_JOB"
echo ""

# Split the input file
echo "Splitting input file..."
TEMP_DATA=$(mktemp)
tail -n +2 "$INPUT_FILE" > "$TEMP_DATA"

START_LINE=1
for ((i=0; i<$NUM_JOBS; i++)); do
    JOB_NUM=$i
    JOB_INPUT_FILE="$SCRATCH_INPUT_DIR/${RUN_NAME}_job${JOB_NUM}_input.csv"
    
    # Calculate lines for this job (distribute remainder across first few jobs)
    CURRENT_LINES=$LINES_PER_JOB
    if [ $i -lt $REMAINDER ]; then
        CURRENT_LINES=$((CURRENT_LINES + 1))
    fi
    
    END_LINE=$((START_LINE + CURRENT_LINES - 1))
    
    # Create split file with header directly on scratch
    echo "$HEADER" > "$JOB_INPUT_FILE"
    sed -n "${START_LINE},${END_LINE}p" "$TEMP_DATA" >> "$JOB_INPUT_FILE"
    
    echo "  Job $JOB_NUM: Lines $START_LINE-$END_LINE ($(wc -l < "$JOB_INPUT_FILE" | tr -d ' ') total including header) -> $JOB_INPUT_FILE"
    
    START_LINE=$((END_LINE + 1))
done

rm "$TEMP_DATA"
echo ""

# Create SLURM scripts
echo "Creating SLURM scripts..."
for ((i=0; i<$NUM_JOBS; i++)); do
    JOB_NUM=$i
    JOB_NAME="${RUN_NAME}_job${JOB_NUM}"
    JOB_INPUT_FILE="$SCRATCH_INPUT_DIR/${RUN_NAME}_job${JOB_NUM}_input.csv"
    SLURM_SCRIPT="$SLURM_DIR/${JOB_NAME}.slurm"
    LOG_FILE="$SCRATCH_LOG_DIR/${JOB_NAME}.log"
    ERROR_FILE="$SCRATCH_LOG_DIR/${JOB_NAME}.err"
    
    cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --account=PAS3230
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${ERROR_FILE}
#SBATCH --time=${SLURM_TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NUM_CORES}

# Print job information
echo "=========================================="
echo "Job: ${JOB_NAME}"
echo "Started: \$(date)"
echo "Running on: \$(hostname)"
echo "=========================================="
echo ""

set -euo pipefail

module load miniconda3
conda activate microlens

# Scratch and final output directories
SCRATCH_CODE_DIR="${SCRATCH_CODE_DIR}"
SCRATCH_OUTPUT_DIR="${SCRATCH_OUTPUT_DIR}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR}"
JOB_INPUT_FILE="${JOB_INPUT_FILE}"

echo "Working directories:"
echo "  Code:       \$SCRATCH_CODE_DIR"
echo "  Output:     \$SCRATCH_OUTPUT_DIR"
echo "  Final:      \$FINAL_OUTPUT_DIR"
echo ""

# Change to scratch code directory
cd \$SCRATCH_CODE_DIR

# Run the detection script with scratch output directory
echo "Running detection script..."
python run_sep_det.py \\
    --run_name ${JOB_NAME} \\
    --input_file \$JOB_INPUT_FILE \\
    --output_dir \$SCRATCH_OUTPUT_DIR \\
    --num_cores ${NUM_CORES} \\
    ${MAG_PLOT_FLAG}

PYTHON_EXIT_CODE=\$?

if [ \$PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python script failed with exit code \$PYTHON_EXIT_CODE"
    exit \$PYTHON_EXIT_CODE
fi

echo ""
echo "Python script completed successfully."
echo ""

# Move outputs from scratch to final destination
echo "Moving outputs to final destination..."
echo "  From: \$SCRATCH_OUTPUT_DIR"
echo "  To:   \$FINAL_OUTPUT_DIR"

# Move all job-related directories and files
for item in \$SCRATCH_OUTPUT_DIR/${JOB_NAME}*; do
    if [ -e "\$item" ]; then
        mv "\$item" "\$FINAL_OUTPUT_DIR/"
        echo "  Moved: \$(basename \$item)"
    fi
done

echo ""
echo "=========================================="
echo "Job completed: \$(date)"
echo "Exit code: 0"
echo "=========================================="

exit 0
EOF
    
    chmod +x "$SLURM_SCRIPT"
    echo "  Created: $SLURM_SCRIPT"
done

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Directory structure on scratch:"
echo "  $SCRATCH_RUN_DIR/"
echo "  ├── outputs/       (job outputs during execution)"
echo "  ├── logs/          (SLURM .log and .err files)"
echo "  ├── code/          (copy of Python scripts)"
echo "  └── split_inputs/  (split input CSV files)"
echo ""
echo "Final outputs will be moved to:"
echo "  $FINAL_OUTPUT_DIR"
echo ""
echo "To submit all jobs, run:"
echo "  cd $SLURM_DIR"
echo "  for script in ${RUN_NAME}_job*.slurm; do sbatch \$script; done"
echo ""
echo "Or submit individual jobs:"
echo "  sbatch $SLURM_DIR/${RUN_NAME}_job0.slurm"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  $SCRATCH_LOG_DIR/${RUN_NAME}_job*.log"
echo "  $SCRATCH_LOG_DIR/${RUN_NAME}_job*.err"
echo "=========================================="
