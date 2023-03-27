#!/bin/bash
#SBATCH --account=PAS1309
jid1=$(sbatch Harmonize.sh)
jid2=$(sbatch --dependency=afterok:$jid1 PrepRFtraining.sh --account=PAS1309)
jid3=$(sbatch --dependency=afterok:$jid2 RF_traning.sh)
jid4=$(sbatch --dependency=afterok:$jid3 RF_forward.sh)
