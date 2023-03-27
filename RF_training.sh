#!/bin/bash
#SBATCH --time=00:30:00             # total run time limit (HH:MM:SS)
#SBATCH --ntasks=1                  #
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=tr_RF     # create a short name for your job
#SBATCH --array=1
#SBATCH --account=PAS1309           # account name
#SBATCH --output=JobInfo/%x_%a.out  # out message
#SBATCH --error=JobInfo/%x_%a.err   # error message

#python RF_training.py
python RF_label_train.py
