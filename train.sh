#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="NetSec Train"
#SBATCH --error="./new_logs/job-%j-netsec_train_script.err"
#SBATCH --output="./new_logs/job-%j-netsec_train_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="hplp"


module purge &&
module load miniforge  &&
source /home/bue6zr/.bashrc  &&
echo "$HOSTNAME" &&
conda deactivate &&
conda activate netsec2
mkdir results/run_${SLURM_JOB_ID}/

PYTHONPATH=. python3 analyzer/main.py --input analyzer/sample_data/ --output results/run_${SLURM_JOB_ID}/test.json

echo "Done" &&
exit