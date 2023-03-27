#!/bin/bash
#SBATCH --time=00:10:00             # total run time limit (HH:MM:SS)
#SBATCH --ntasks=1                  #
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-41
#SBATCH --job-name=harmonize     # create a short name for your job
#SBATCH --account=PAS1309           # account name
#SBATCH --output=JobInfo/%x_%a.out  # out message
#SBATCH --error=JobInfo/%x_%a.err   # error message

python Harmonize.py
