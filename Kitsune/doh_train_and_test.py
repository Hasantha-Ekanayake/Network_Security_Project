#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import time
from multiprocessing import Process

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from Kitsune import Kitsune


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_kitsune(path, packet_limit, max_ae, fm_grace, ad_grace):
    return Kitsune(
        path,
        packet_limit,
        max_ae,
        fm_grace,
        ad_grace
    )


def get_anomaly_detector(kitsune_obj):
    if hasattr(kitsune_obj, "AnomDetector"):
        return kitsune_obj.AnomDetector
    if hasattr(kitsune_obj, "AD"):
        return kitsune_obj.AD
    raise AttributeError("Could not find Kitsune anomaly detector object.")


def set_anomaly_detector(kitsune_obj, detector):
    if hasattr(kitsune_obj, "AnomDetector"):
        kitsune_obj.AnomDetector = detector
        return
    if hasattr(kitsune_obj, "AD"):
        kitsune_obj.AD = detector
        return
    raise AttributeError("Could not set Kitsune anomaly detector object.")

def plot_rmse_distribution(rmse_scores, output_path, title):
    plt.figure(figsize=(8, 5))

    plt.hist(
        rmse_scores,
        bins=200,
        density=True,
        alpha=0.7,
    )

    plt.xlabel("RMSE")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_log_rmse_distribution(rmse_scores, output_path, title):
    rmse_scores = rmse_scores[rmse_scores > 0]

    plt.figure(figsize=(8, 5))

    plt.hist(
        np.log(rmse_scores),
        bins=200,
        density=True,
        alpha=0.7,
    )

    plt.xlabel("log(RMSE)")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_rmse_overlay(clean_rmse, malicious_rmse, output_path):
    plt.figure(figsize=(8, 5))

    plt.hist(clean_rmse, bins=200, density=True, alpha=0.5, label="Clean")
    plt.hist(malicious_rmse, bins=200, density=True, alpha=0.5, label="Malicious")

    plt.xlabel("RMSE")
    plt.ylabel("Density")
    plt.title("RMSE Distribution: Clean vs Malicious")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def train_kitsune(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Training Kitsune")
    print("=" * 80)

    K = build_kitsune(
        path=args.train_pcap,
        packet_limit=args.packet_limit,
        max_ae=args.max_ae,
        fm_grace=args.fm_grace,
        ad_grace=args.ad_grace,
    )

    rmse_scores = []
    start_time = time.time()

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    i = 0
    while True:
        i += 1

        rmse = K.proc_next_packet()

        if rmse == -1:
            print("End of training PCAP reached.")
            break

        rmse_scores.append(float(rmse))

        if i % args.print_every == 0:
            print(f"Processed {i} packets | RMSE = {rmse}")

        if args.checkpoint_every > 0 and i % args.checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"kitsune_checkpoint_{i}.pkl")
            save_pickle(K, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

        if args.max_train_packets > 0 and i >= args.max_train_packets:
            print("Reached max_train_packets.")
            break

    train_time = time.time() - start_time

    final_model_path = os.path.join(args.output_dir, "kitsune_model.pkl")
    detector_path = os.path.join(args.output_dir, "kitsune_detector.pkl")
    rmse_path = os.path.join(args.output_dir, "train_rmse.npy")

    save_pickle(K, final_model_path)
    save_pickle(get_anomaly_detector(K), detector_path)
    np.save(rmse_path, np.asarray(rmse_scores))

    threshold_info = compute_threshold(
        rmse_scores=np.asarray(rmse_scores),
        fm_grace=args.fm_grace,
        ad_grace=args.ad_grace,
        percentile=args.threshold_percentile,
    )

    threshold_path = os.path.join(args.output_dir, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump(threshold_info, f, indent=4)

    plot_train_rmse(
        rmse_scores=np.asarray(rmse_scores),
        threshold=threshold_info["threshold_rmse"],
        output_path=os.path.join(args.output_dir, "train_rmse_curve.png"),
        fm_grace=args.fm_grace,
        ad_grace=args.ad_grace,
    )

    plot_rmse_distribution(
        np.asarray(rmse_scores),
        os.path.join(args.output_dir, "train_rmse_distribution.png"),
        "Training RMSE Distribution"
    )

    plot_log_rmse_distribution(
        np.asarray(rmse_scores),
        os.path.join(args.output_dir, "train_log_rmse_distribution.png"),
        "Training Log-RMSE Distribution"
    )

    summary = {
        "train_pcap": args.train_pcap,
        "num_packets_processed": int(len(rmse_scores)),
        "train_time_sec": float(train_time),
        "max_ae": int(args.max_ae),
        "fm_grace": int(args.fm_grace),
        "ad_grace": int(args.ad_grace),
        "model_path": final_model_path,
        "detector_path": detector_path,
        "threshold_path": threshold_path,
        "threshold_info": threshold_info,
    }

    summary_path = os.path.join(args.output_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("Training complete.")
    print("Saved model:", final_model_path)
    print("Saved detector:", detector_path)
    print("Saved threshold:", threshold_path)

    return detector_path, threshold_info["threshold_rmse"]


def compute_threshold(rmse_scores, fm_grace, ad_grace, percentile):
    execution_start = fm_grace + ad_grace

    if len(rmse_scores) <= execution_start + 10:
        sample = rmse_scores
    else:
        sample = rmse_scores[execution_start:]

    sample = sample[np.isfinite(sample)]
    sample = sample[sample > 0]

    threshold = float(np.percentile(sample, percentile))

    log_sample = np.log(sample)
    log_mean = float(np.mean(log_sample))
    log_std = float(np.std(log_sample))

    return {
        "threshold_rmse": threshold,
        "threshold_percentile": float(percentile),
        "num_threshold_samples": int(len(sample)),
        "log_rmse_mean": log_mean,
        "log_rmse_std": log_std,
    }


def plot_train_rmse(rmse_scores, threshold, output_path, fm_grace, ad_grace):
    plt.figure(figsize=(10, 5))
    plt.plot(rmse_scores, linewidth=0.5)
    plt.axhline(threshold, linestyle="--", label=f"Threshold = {threshold:.6f}")
    plt.axvline(fm_grace, linestyle="--", label="Feature mapping ends")
    plt.axvline(fm_grace + ad_grace, linestyle="--", label="AD training ends")
    plt.yscale("log")
    plt.xlabel("Packet index")
    plt.ylabel("RMSE")
    plt.title("Kitsune Training RMSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def test_kitsune_single(
    pcap_path,
    label_name,
    detector_path,
    threshold_rmse,
    output_dir,
    packet_limit,
    max_ae,
    buffer_packets,
    print_every,
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"Testing Kitsune on {label_name}")
    print("=" * 80)

    detector = load_pickle(detector_path)

    K = build_kitsune(
        path=pcap_path,
        packet_limit=packet_limit,
        max_ae=max_ae,
        fm_grace=0,
        ad_grace=0,
    )

    set_anomaly_detector(K, detector)

    rmse_scores = []
    preds = []

    i = 0
    start_time = time.time()

    while True:
        i += 1

        rmse = K.proc_next_packet()

        if rmse == -1:
            break

        rmse = float(rmse)
        rmse_scores.append(rmse)

        if i <= buffer_packets:
            pred = 0
        else:
            pred = int(rmse > threshold_rmse)

        preds.append(pred)

        if i % print_every == 0:
            print(f"[{label_name}] Processed {i} packets | RMSE = {rmse}")

    elapsed = time.time() - start_time

    rmse_scores = np.asarray(rmse_scores)
    preds = np.asarray(preds)

    np.save(os.path.join(output_dir, f"{label_name}_rmse.npy"), rmse_scores)
    np.save(os.path.join(output_dir, f"{label_name}_preds.npy"), preds)

    num_eval_packets = max(0, len(preds) - buffer_packets)
    num_anomalies = int(np.sum(preds[buffer_packets:])) if len(preds) > buffer_packets else 0
    anomaly_fraction = float(num_anomalies / num_eval_packets) if num_eval_packets > 0 else 0.0

    summary = {
        "label_name": label_name,
        "pcap_path": pcap_path,
        "num_packets": int(len(rmse_scores)),
        "buffer_packets": int(buffer_packets),
        "num_eval_packets": int(num_eval_packets),
        "num_anomalous_packets": int(num_anomalies),
        "anomaly_fraction": anomaly_fraction,
        "threshold_rmse": float(threshold_rmse),
        "elapsed_sec": float(elapsed),
        "rmse_mean": float(np.mean(rmse_scores)) if len(rmse_scores) else None,
        "rmse_median": float(np.median(rmse_scores)) if len(rmse_scores) else None,
        "rmse_p95": float(np.percentile(rmse_scores, 95)) if len(rmse_scores) else None,
        "rmse_p99": float(np.percentile(rmse_scores, 99)) if len(rmse_scores) else None,
    }

    with open(os.path.join(output_dir, f"{label_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    plot_test_rmse(
        rmse_scores=rmse_scores,
        preds=preds,
        threshold=threshold_rmse,
        buffer_packets=buffer_packets,
        output_path=os.path.join(output_dir, f"{label_name}_rmse_curve.png"),
        title=f"Kitsune RMSE: {label_name}",
    )

    plot_rmse_distribution(
        rmse_scores,
        os.path.join(output_dir, f"{label_name}_rmse_distribution.png"),
        f"RMSE Distribution: {label_name}"
    )

    plot_log_rmse_distribution(
        rmse_scores,
        os.path.join(output_dir, f"{label_name}_log_rmse_distribution.png"),
        f"Log RMSE Distribution: {label_name}"
    )

    print(f"[{label_name}] Test complete.")
    print(f"[{label_name}] Anomaly fraction:", anomaly_fraction)


def plot_test_rmse(rmse_scores, preds, threshold, buffer_packets, output_path, title):
    plt.figure(figsize=(10, 5))

    x = np.arange(len(rmse_scores))
    plt.plot(x, rmse_scores, linewidth=0.5, label="RMSE")
    plt.axhline(threshold, linestyle="--", label=f"Threshold = {threshold:.6f}")

    if buffer_packets > 0:
        plt.axvline(buffer_packets, linestyle="--", label="Buffer ends")

    anomaly_idx = np.where(preds == 1)[0]
    if len(anomaly_idx) > 0:
        plt.scatter(
            anomaly_idx,
            rmse_scores[anomaly_idx],
            s=2,
            label="Flagged anomalous",
        )

    plt.yscale("log")
    plt.xlabel("Packet index")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_tests_parallel(args, detector_path, threshold_rmse):
    clean_output_dir = os.path.join(args.output_dir, "test_clean")
    malicious_output_dir = os.path.join(args.output_dir, "test_malicious")

    p1 = Process(
        target=test_kitsune_single,
        kwargs={
            "pcap_path": args.clean_test_pcap,
            "label_name": "clean",
            "detector_path": detector_path,
            "threshold_rmse": threshold_rmse,
            "output_dir": clean_output_dir,
            "packet_limit": args.test_packet_limit,
            "max_ae": args.max_ae,
            "buffer_packets": args.buffer_packets,
            "print_every": args.print_every,
        },
    )

    p2 = Process(
        target=test_kitsune_single,
        kwargs={
            "pcap_path": args.malicious_test_pcap,
            "label_name": "malicious",
            "detector_path": detector_path,
            "threshold_rmse": threshold_rmse,
            "output_dir": malicious_output_dir,
            "packet_limit": args.test_packet_limit,
            "max_ae": args.max_ae,
            "buffer_packets": args.buffer_packets,
            "print_every": args.print_every,
        },
    )

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    if p1.exitcode != 0:
        raise RuntimeError("Clean test process failed.")
    if p2.exitcode != 0:
        raise RuntimeError("Malicious test process failed.")

    clean_rmse = np.load(os.path.join(clean_output_dir, "clean_rmse.npy"))
    mal_rmse = np.load(os.path.join(malicious_output_dir, "malicious_rmse.npy"))

    plot_rmse_overlay(
        clean_rmse,
        mal_rmse,
        os.path.join(args.output_dir, "rmse_overlay_clean_vs_malicious.png")
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_pcap", required=True)
    parser.add_argument("--clean_test_pcap", required=True)
    parser.add_argument("--malicious_test_pcap", required=True)

    parser.add_argument("--output_dir", default="kitsune_results")

    parser.add_argument("--packet_limit", type=float, default=np.inf)
    parser.add_argument("--test_packet_limit", type=float, default=np.inf)

    parser.add_argument("--max_ae", type=int, default=10)
    parser.add_argument("--fm_grace", type=int, default=5000)
    parser.add_argument("--ad_grace", type=int, default=50000)

    parser.add_argument("--max_train_packets", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=10000)
    parser.add_argument("--print_every", type=int, default=1000)

    parser.add_argument("--threshold_percentile", type=float, default=99.0)
    parser.add_argument("--buffer_packets", type=int, default=1000)

    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training and reuse detector_path + threshold_path",
    )

    parser.add_argument("--detector_path", default=None)
    parser.add_argument("--threshold_path", default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.skip_train:
        if args.detector_path is None or args.threshold_path is None:
            raise ValueError("--skip_train requires --detector_path and --threshold_path")

        detector_path = args.detector_path

        with open(args.threshold_path, "r") as f:
            threshold_info = json.load(f)

        threshold_rmse = threshold_info["threshold_rmse"]

    else:
        detector_path, threshold_rmse = train_kitsune(args)

    run_tests_parallel(args, detector_path, threshold_rmse)

    print("All Kitsune training/testing complete.")


if __name__ == "__main__":
    main()