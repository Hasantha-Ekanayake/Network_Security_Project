#!/bin/bash

#SBATCH --job-name="Time AE Test"
#SBATCH --error="./logs/time_ae_test/job-%j-time_ae_test.err"
#SBATCH --output="./logs/time_ae_test/job-%j-time_ae_test.out"
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

mkdir -p results/time_ae_test/run_${SLURM_JOB_ID}

export PYTHONPATH=.

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Change this to the training job id you want to test
TRAIN_JOB_ID=12263455

for w in 4 5 6 7 8 9 10; do
    python analyzer/main_time_test.py \
        --experiment_dir results/time_ae_train/run_${TRAIN_JOB_ID}/window_${w} \
        --batch_size 128 \
        --min_anomaly_fraction 0.30 \
        --min_consecutive_anomalies 3
done

echo "Testing done" &&
exit