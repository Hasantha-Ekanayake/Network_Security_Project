#!/bin/bash

#SBATCH --job-name="Stat AE Train"
#SBATCH --error="./logs/stat_ae_train/job-%j-stat_ae_train.err"
#SBATCH --output="./logs/stat_ae_train/job-%j-stat_ae_train.out"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:1
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

mkdir -p results/stat_ae_train/run_${SLURM_JOB_ID}

export PYTHONPATH=.

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

BATCH_SIZE=128

python analyzer/main_stat_train.py \
    --nondoh ../dataset/CSVs/l1-nondoh.csv \
    --benign ../dataset/CSVs/l2-benign.csv \
    --malicious ../dataset/CSVs/l2-malicious.csv \
    --output_dir results/stat_ae_train/run_${SLURM_JOB_ID} \
    --clean_labels Benign \
    --epochs 200 \
    --batch_size ${BATCH_SIZE} \
    --latent_dim 16 \
    --random_state 42 \
    --nondoh_ratio 0.0

python analyzer/main_stat_test.py \
    --experiment_dir results/stat_ae_train/run_${SLURM_JOB_ID} \
    --model_name best_epoch_model.keras \
    --batch_size ${BATCH_SIZE}

echo "Training done" &&
exit