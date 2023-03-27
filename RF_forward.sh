#!/bin/bash
#SBATCH --time=48:00:00             # total run time limit (HH:MM:SS)
#SBATCH --ntasks=1                  #
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-16
#SBATCH --job-name=forward     # create a short name for your job
#SBATCH --account=PAS1309           # account name
#SBATCH --output=JobInfo/%x_%a.out  # out message
#SBATCH --error=JobInfo/%x_%a.err   # error message

python RF_forward.py
