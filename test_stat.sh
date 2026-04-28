#!/bin/bash

#SBATCH --job-name="Stat AE Test"
#SBATCH --error="./logs/stat_ae_test/job-%j-stat_ae_test.err"
#SBATCH --output="./logs/stat_ae_test/job-%j-stat_ae_test.out"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="hplp"

module purge
module load miniforge

source /home/bue6zr/.bashrc

echo "Running on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

conda deactivate
conda activate netsec3

mkdir -p results/stat_ae_test/run_${SLURM_JOB_ID}

export PYTHONPATH=.

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Change this to the training run directory you want to test
EXPERIMENT_DIR="results/stat_ae_train/run_12281958"

python analyzer/main_stat_test.py \
    --experiment_dir ${EXPERIMENT_DIR} \
    --model_name best_epoch_model.keras \
    --batch_size 128

echo "Testing done" &&
exit