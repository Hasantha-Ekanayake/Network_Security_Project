#!/bin/bash

#SBATCH --job-name="Kitsune Train"
#SBATCH --error="./logs/kitsune_train/job-%j-kitsune_train.err"
#SBATCH --output="./logs/kitsune_train/job-%j-kitsune_train.out"
#SBATCH --partition="standard"
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
conda activate netsec_kitsune

mkdir -p results/kitsune/run_${SLURM_JOB_ID}

export PYTHONPATH=.

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

python Kitsune/doh_train_and_test.py \
    --train_pcap ../dataset/PCAPs/DoHBenign-NonDoH/Cloudflare/dump_00001_20200113152847.pcap \
    --clean_test_pcap ../dataset/PCAPs/DoHBenign-NonDoH/Cloudflare/dump_00002_20200113162614.pcap \
    --malicious_test_pcap ../dataset/PCAPs/MaliciousDoH/dns2tcp_tunnel_1111_doh1_2020-03-31T21:54:32.055088.pcap \
    --output_dir results/kitsune/run_${SLURM_JOB_ID} \
    --max_ae 10 \
    --packet_limit 200000 \
    --fm_grace 5000 \
    --ad_grace 95000 \
    --checkpoint_every 10000 \
    --threshold_percentile 99.0 \
    --buffer_packets 1000 \
    --print_every 1000


echo "Training done" &&
exit