#!/bin/bash

#SBATCH --job-name="Stat AE Test"
#SBATCH --error="./logs/stat_ae_test/job-%j-stat_ae_test.err"
#SBATCH --output="./logs/stat_ae_test/job-%j-stat_ae_test.out"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="shukla-lab"

module purge
module load miniforge

source /home/uyq6nu/.bashrc

echo "Running on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

conda deactivate
conda activate net_sec_env

mkdir -p results/stat_ae_test/run_${SLURM_JOB_ID}

export PYTHONPATH=.

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Change this to the training run directory you want to test
EXPERIMENT_DIR="results/stat_ae_train/run_12409366"

python analyzer/main_stat_test.py \
    --experiment_dir ${EXPERIMENT_DIR} \
    --model_name best_epoch_model.keras \
    --batch_size 128

echo "Testing done" &&
exit