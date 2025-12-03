# SLURM Job Submission Scripts

This directory contains scripts for splitting input files and creating SLURM job scripts for parallel execution of `run_sep_det.py`.

## Scripts

### 1. `split_and_submit.sh`
Main script that splits an input CSV file into multiple smaller files and creates corresponding SLURM job scripts.

**Usage:**
```bash
./split_and_submit.sh --input_file <file> --num_jobs <n> --run_name <name> --output_dir <dir> [OPTIONS]
```

**Required Arguments:**
- `--input_file`: Path to input CSV file with system parameters
- `--num_jobs`: Number of jobs to split into
- `--run_name`: Global run name (individual jobs will be named `<run_name>_job0`, `<run_name>_job1`, etc.)
- `--output_dir`: Output directory for results (shared across all jobs)

**Optional Arguments:**
- `--num_cores`: Number of cores per job (default: 1)
- `--mag_plot`: Flag to plot magnification maps
- `--slurm_time`: SLURM time limit (default: 24:00:00)
- `--slurm_mem`: SLURM memory per job (default: 16G)
- `--slurm_partition`: SLURM partition (default: normal)

**Example:**
```bash
./split_and_submit.sh \
    --input_file ../oom_detectability/input.csv \
    --num_jobs 10 \
    --run_name test_run \
    --output_dir ../oom_detectability/results \
    --num_cores 4 \
    --mag_plot \
    --slurm_time 48:00:00 \
    --slurm_mem 32G
```

### 2. `submit_all_jobs.sh`
Helper script to submit all SLURM jobs for a given run.

**Usage:**
```bash
./submit_all_jobs.sh <run_name>
```

**Example:**
```bash
./submit_all_jobs.sh test_run
```

### 3. `check_job_status.sh`
Utility script to check the status of all jobs for a given run.

**Usage:**
```bash
./check_job_status.sh <run_name>
```

**Example:**
```bash
./check_job_status.sh test_run
```

Shows:
- Jobs currently in the SLURM queue
- Completion status of each job
- Recent errors from error logs
- Exit codes for completed jobs

### 4. `cancel_all_jobs.sh`
Utility script to cancel all running/pending jobs for a given run.

**Usage:**
```bash
./cancel_all_jobs.sh <run_name>
```

**Example:**
```bash
./cancel_all_jobs.sh test_run
```

This script will prompt for confirmation before cancelling jobs.

## Workflow

1. **Prepare your input CSV file** with the following columns:
   - `s2`, `q2`, `s3`, `q3`, `psi`, `rho`, `tE`, `cad`, `bin_box`, `planet_box`, `contour_threshold`, `alpha_density`, `n_pot`

2. **Run the split script** to create job files and SLURM scripts:
   ```bash
   cd scripts
   ./split_and_submit.sh \
       --input_file ../oom_detectability/input.csv \
       --num_jobs 5 \
       --run_name my_run \
       --output_dir ../oom_detectability/output \
       --num_cores 8
   ```

3. **Submit all jobs**:
   ```bash
   ./submit_all_jobs.sh my_run
   ```
   
   Or submit individual jobs manually:
   ```bash
   cd ../slurm_scripts
   sbatch my_run_job0.slurm
   sbatch my_run_job1.slurm
   # etc.
   ```

4. **Monitor jobs**:
   ```bash
   # Check status of all jobs for your run
   ./check_job_status.sh my_run
   
   # Or use SLURM directly
   squeue -u $USER
   ```

5. **Check logs**:
   ```bash
   # View live log
   tail -f ../slurm_scripts/my_run_job0.log
   
   # Check for errors
   cat ../slurm_scripts/my_run_job0.err
   ```

6. **Cancel jobs if needed**:
   ```bash
   ./cancel_all_jobs.sh my_run
   ```

## Output Structure

After running the split script, the following structure is created:

```
slurm_scripts/
├── split_inputs/
│   ├── my_run_job0_input.csv
│   ├── my_run_job1_input.csv
│   └── ...
├── my_run_job0.slurm
├── my_run_job0.log
├── my_run_job0.err
├── my_run_job1.slurm
├── my_run_job1.log
├── my_run_job1.err
└── ...
```

All jobs write results to the shared output directory specified with `--output_dir`. Each job creates subdirectories named `<run_name>_job<N>_<system_index>` for individual system results.

## Notes

- **Module loading**: Edit the SLURM scripts in `slurm_scripts/` to load necessary modules for your cluster (e.g., `module load python/3.9`)
- **Virtual environments**: Uncomment and modify the virtual environment activation line in the SLURM scripts if needed
- **SLURM settings**: Adjust time, memory, and partition settings based on your cluster's configuration and your job requirements
- **Job dependencies**: If you need jobs to run sequentially or have dependencies, use SLURM's `--dependency` flag when submitting
- **Shared output**: All jobs write to the same output directory, so make sure it's accessible from all compute nodes

## Troubleshooting

- **Jobs fail immediately**: Check the `.err` files in `slurm_scripts/`
- **Module not found**: Make sure to load necessary modules in the SLURM script
- **Permission denied**: Ensure scripts are executable (`chmod +x script.sh`)
- **Input file format**: Verify your input CSV has the correct header and column format

