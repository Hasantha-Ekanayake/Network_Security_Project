#!/bin/bash

#SBATCH --job-name="Time AE Train"
#SBATCH --error="./logs/time_ae_train/job-%j-time_ae_train.err"
#SBATCH --output="./logs/time_ae_train/job-%j-time_ae_train.out"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --account="hplp"

module purge
module load miniforge

source /home/bue6zr/.bashrc

echo "Running on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

conda deactivate
conda activate netsec3

mkdir -p results/time_ae_train/run_${SLURM_JOB_ID}

export PYTHONPATH=.

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

BATCH_SIZE=128

python analyzer/main_time_train.py \
    --nondoh processed_data/time_series/non_doh.json \
    --benign processed_data/time_series/benign_doh.json \
    --malicious processed_data/time_series/malicious_doh.json \
    --output_dir results/time_ae_train/run_${SLURM_JOB_ID} \
    --clean_labels NonDoH Benign \
    --window_min 4 \
    --window_max 8 \
    --epochs 200 \
    --batch_size ${BATCH_SIZE} \
    --latent_dim 4 \
    --random_state 42 \
    --lstm_units 16 \
    --patience 20

for w in 4 5 6 7 8; do
    python analyzer/main_time_test.py \
        --experiment_dir results/time_ae_train/run_${SLURM_JOB_ID}/window_${w} \
        --batch_size ${BATCH_SIZE} \
        --min_anomaly_fraction 0.95
done

echo "Training done" &&
exit