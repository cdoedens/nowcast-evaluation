import subprocess
from pathlib import Path
import pandas as pd
import sys, os

script_to_run = "calculate-metrics"
log_dir = f"/home/548/cd3022/repos/nowcast-evaluation/outputs/logs/{script_to_run}"
LOG_PATH = Path(log_dir)
LOG_PATH.mkdir(parents=True, exist_ok=True)

num_months = 12
days_per_batch = 1
dates = []


for date in pd.date_range("2025-01-01", "2025-12-31"):
    DATE = f"{date.year}-{date.month}-{date.day}"
    # Generate a unique file name based on iteration
    joboutdir = '/home/548/cd3022/repos/nowcast-evaluation/scripts/pbs/'
    job_script_filename = joboutdir + f'{script_to_run}___{DATE}.qsub'
    
    # Open the file for writing
    with open(job_script_filename, "w") as f3:
        f3.write('#!/bin/bash \n')
        f3.write('#PBS -l walltime=1:00:00 \n')
        f3.write('#PBS -l mem=46GB \n')
        f3.write('#PBS -l ncpus=12 \n')
        f3.write('#PBS -l jobfs=10GB \n')
        f3.write('#PBS -l storage=gdata/xp65+gdata/rv74+scratch/er8+gdata/er8 \n')
        f3.write('#PBS -l other=hyperthread \n')
        f3.write('#PBS -q normal \n')
        f3.write('#PBS -P er8 \n')
        f3.write(f'#PBS -o {log_dir}/{DATE}.log \n')
        f3.write('#PBS -j oe \n')
        f3.write('cd /home/548/cd3022/repos/nowcast-evaluation \n') 
        f3.write('source env.sh \n')
        f3.write(f'python3 scripts/{script_to_run}.py --date {DATE}\n')


    # Submit the generated script to the job scheduler (PBS) using qsub
    try:
        # Run the qsub command and submit the script
        subprocess.run(['qsub', job_script_filename], check=True)
        print(f"Successfully submitted job for {script_to_run} with date {DATE}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job script for {script_to_run} with date {DATE}: {e}")