#!/bin/bash

#SBATCH --job-name="Stat AE Train"
#SBATCH --error="./new_logs/stat_ae_train/job-%j-stat_ae_train.err"
#SBATCH --output="./new_logs/stat_ae_train/job-%j-stat_ae_train.out"
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

mkdir -p results/stat_ae_train/run_${SLURM_JOB_ID}

export PYTHONPATH=.

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

python analyzer/main_stat_train.py \
    --nondoh ../dataset/CSVs/l1-nondoh.csv \
    --benign ../dataset/CSVs/l2-benign.csv \
    --malicious ../dataset/CSVs/l2-malicious.csv \
    --output_dir results/stat_ae_train/run_${SLURM_JOB_ID} \
    --epochs 200 \
    --batch_size 512 \
    --latent_dim 8 \
    --random_state 42

echo "Training done" &&
exit