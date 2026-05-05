# Network Security Project: Autoencoder-based Anomaly Detection for Encrypted DoH Traffic

This repository builds on DoHlyzer to detect anomalies in encrypted DNS-over-HTTPS (DoH) traffic using autoencoders. The project extracts statistical and time-series features from the CIRA-CIC-DoHBrw-2020 dataset and evaluates reconstruction-error-based anomaly detection.

## Repository Structure

- `meter/` — extracts statistical and time-series features from PCAP traffic.
- `analyzer/` — trains and tests autoencoder models.
- `visualizer/` — visualizes extracted clump/time-series features.
- `Kitsune/` — additional anomaly detection baseline/resources.
- `train.sh` — general autoencoder training script.
- `train_stat.sh` — trains the statistical-feature autoencoder.
- `train_time.sh` — trains the time-series-feature autoencoder.
- `test_stat.sh` — tests the statistical-feature autoencoder.
- `test_time.sh` — tests the time-series-feature autoencoder.
- `create_json.sh` — creates JSON clump files.
- `create_json_mal.sh` — creates JSON files for malicious traffic.
- `requirements.txt` — original DoHlyzer dependencies.
- `requirement_new.txt` — updated dependencies for the autoencoder implementation.

## Dataset

This work uses the CIRA-CIC-DoHBrw-2020 dataset, which contains Non-DoH, benign DoH, and malicious DoH traffic.

## Method

The workflow consists of:

1. Extracting flow-level statistical features and time-series clump features.
2. Training autoencoders on clean traffic.
3. Computing reconstruction RMSE.
4. Using reconstruction error as an anomaly score to detect malicious DoH traffic.

## Citation

If you use this repository, please cite the original DoHlyzer paper, the dataset, DoHlyzer, and this repository.

## Project Team members

* **Kavish Ranawella:** Conducted the literature review, designed and implemented the autoencoder models, and contributed to result generation and analysis.  
* **Hasantha Ekanayake:** Conducted the literature review, performed dataset preprocessing, and contributed to result generation and analysis.


