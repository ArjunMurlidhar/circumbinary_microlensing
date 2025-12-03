# Quick Start Guide

## TL;DR - Run Jobs in 3 Steps

```bash
cd scripts

# 1. Split input and create SLURM scripts
./split_and_submit.sh \
    --input_file ../oom_detectability/input.csv \
    --num_jobs 10 \
    --run_name my_run \
    --output_dir ../oom_detectability/output \
    --num_cores 8

# 2. Submit all jobs
./submit_all_jobs.sh my_run

# 3. Check status
./check_job_status.sh my_run
```

## Common Commands

### Setup and Submit
```bash
# Basic usage with minimal options
./split_and_submit.sh \
    --input_file INPUT.csv \
    --num_jobs 5 \
    --run_name run1 \
    --output_dir ./output

# With all options
./split_and_submit.sh \
    --input_file INPUT.csv \
    --num_jobs 10 \
    --run_name run1 \
    --output_dir ./output \
    --num_cores 8 \
    --mag_plot \
    --slurm_time 48:00:00 \
    --slurm_mem 32G \
    --slurm_partition compute
```

### Monitoring
```bash
# Check job status
./check_job_status.sh my_run

# Watch queue
watch -n 10 'squeue -u $USER'

# View live logs
tail -f ../slurm_scripts/my_run_job0.log
```

### Management
```bash
# Cancel all jobs for a run
./cancel_all_jobs.sh my_run

# Cancel specific job
scancel JOB_ID

# Hold/release jobs
scontrol hold JOB_ID
scontrol release JOB_ID
```

## Input File Format

Your CSV file should have these columns (with header):
```
s2,q2,s3,q3,psi,rho,tE,cad,bin_box,planet_box,contour_threshold,alpha_density,n_pot
```

Example:
```csv
s2,q2,s3,q3,psi,rho,tE,cad,bin_box,planet_box,contour_threshold,alpha_density,n_pot
1.25,0.001,1.35,0.0001,0,0.01,100,1.0,0.5,0.3,0.05,360,5
1.25,0.001,1.35,0.0001,72,0.01,100,1.0,0.5,0.3,0.05,360,5
```

## Output Structure

```
output_dir/
├── my_run_job0_0/     # Results for job 0, system 0
│   ├── bin_map.fits
│   ├── planet_map.fits
│   └── ...
├── my_run_job0_1/     # Results for job 0, system 1
└── ...

slurm_scripts/
├── split_inputs/
│   ├── my_run_job0_input.csv
│   └── ...
├── my_run_job0.slurm
├── my_run_job0.log
├── my_run_job0.err
└── ...
```

## Typical Workflow

1. **Prepare** your input CSV file
2. **Split** into parallel jobs: `./split_and_submit.sh ...`
3. **Edit** SLURM scripts if needed (load modules, set environment)
4. **Submit** jobs: `./submit_all_jobs.sh my_run`
5. **Monitor** progress: `./check_job_status.sh my_run`
6. **Check** results in output directory
7. **Combine** results from `my_run_results.csv` in each job directory

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Jobs fail immediately | Check `.err` files in `slurm_scripts/` |
| Module not found | Edit `.slurm` files to load required modules |
| Out of memory | Increase `--slurm_mem` parameter |
| Jobs pending forever | Check partition availability with `sinfo` |
| Permission denied | Run `chmod +x scripts/*.sh` |

## Tips

- Start with a **small test run** (2-3 jobs) to verify everything works
- Adjust **num_jobs** based on cluster availability and total systems
- Use **--mag_plot** sparingly (creates many plots, slower)
- Check **cluster limits** for max memory, time, and cores
- Results from all jobs are **independent** and can be combined later
- Use **job arrays** for very large runs (contact your cluster admin)

For detailed information, see [README.md](README.md)

