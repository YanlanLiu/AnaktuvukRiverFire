#!/bin/bash
#SBATCH --time=00:10:00             # total run time limit (HH:MM:SS)
#SBATCH --ntasks=1                 #
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-16
#SBATCH --job-name=plot     # create a short name for your job
#SBATCH --account=PAS1309           # account name
#SBATCH --output=JobInfo/%x_%a.out  # out message
#SBATCH --error=JobInfo/%x_%a.err   # error message

python PlotLC.py
