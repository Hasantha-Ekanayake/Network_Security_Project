#!/usr/bin/env python3

import argparse
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)


def compute_segment_rmse(x, x_recon):
    return np.sqrt(np.mean((x - x_recon) ** 2, axis=(1, 2)))


def plot_roc(y_true, scores, output_path):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.5f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Segment-Level ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return roc_auc, threshold, fpr[optimal_idx], tpr[optimal_idx]


def plot_precision_recall(y_true, scores, output_path):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.5f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Segment-Level Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return pr_auc

def plot_rmse_histogram_by_class(y_true, scores, output_path):
    plt.figure(figsize=(8, 5))

    clean_scores = scores[y_true == 0]
    malicious_scores = scores[y_true == 1]

    plt.hist(clean_scores, bins=100, alpha=0.5, density=True, label="Clean")
    plt.hist(malicious_scores, bins=100, alpha=0.5, density=True, label="Malicious")

    plt.xlabel("Segment Reconstruction RMSE")
    plt.ylabel("Density")
    plt.title("Segment RMSE Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rmse_histogram_by_subclass(y_original, scores, output_path):
    plt.figure(figsize=(8, 5))

    for label in sorted(set(y_original)):
        mask = y_original == label
        plt.hist(
            scores[mask],
            bins=100,
            alpha=0.5,
            density=True,
            label=label,
        )

    plt.xlabel("Segment Reconstruction RMSE")
    plt.ylabel("Density")
    plt.title("Segment RMSE Distribution by True Class")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, x_labels, y_labels, output_path, title):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), y_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            plt.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
            )

    plt.colorbar(label="Row-normalized percentage")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_custom_confusion_matrix(y_original, y_pred):
    true_labels = list(dict.fromkeys(y_original))
    pred_labels = ["Clean", "Malicious"]

    cm = np.zeros((len(true_labels), len(pred_labels)), dtype=int)

    for true_label, pred in zip(y_original, y_pred):
        pred_label = "Malicious" if pred == 1 else "Clean"
        row = true_labels.index(true_label)
        col = pred_labels.index(pred_label)
        cm[row, col] += 1

    return cm, true_labels, pred_labels


def aggregate_segments_to_flows(
    segment_scores,
    segment_preds,
    segment_labels,
    flow_ids,
    y_test_original,
    min_anomaly_fraction=0.30,
):
    flow_results = {}

    for fid in np.unique(flow_ids):
        mask = flow_ids == fid

        scores = segment_scores[mask]
        preds = segment_preds[mask]
        labels = segment_labels[mask]
        orig_labels = y_test_original[mask]

        true_label = int(np.max(labels))
        true_original = str(orig_labels[0])

        anomaly_fraction = float(np.mean(preds))
        max_score = float(np.max(scores))
        mean_score = float(np.mean(scores))

        flow_pred = int(anomaly_fraction >= min_anomaly_fraction)

        flow_results[int(fid)] = {
            "true_label": true_label,
            "true_original": true_original,
            "num_segments": int(len(scores)),
            "max_score": max_score,
            "mean_score": mean_score,
            "anomaly_fraction": anomaly_fraction,
            "pred_fraction_rule": flow_pred,
        }

    return flow_results


def flow_results_to_arrays(flow_results):
    y_true = []
    y_pred = []
    y_original = []

    for _, r in flow_results.items():
        y_true.append(r["true_label"])
        y_pred.append(r["pred_fraction_rule"])
        y_original.append(r["true_original"])

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(y_original, dtype=object),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--model_name", default="best_epoch_model.keras")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--min_anomaly_fraction",
        type=float,
        default=0.30,
        help="Flow is malicious if this fraction of its segments are anomalous",
    )

    args = parser.parse_args()

    split_path = os.path.join(args.experiment_dir, "data_split.pkl")
    model_path = os.path.join(args.experiment_dir, args.model_name)

    with open(split_path, "rb") as f:
        data = pickle.load(f)

    model = tf.keras.models.load_model(model_path)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"]
    y_test_original = data["y_test_original"]

    # Check how much padding exists
    padding_mask = np.all(X_test[:, :, :4] == -1, axis=2)  # ignore direction
    padding_fraction = padding_mask.mean()

    print(f"Padding fraction (all timesteps): {padding_fraction:.4f}")

    # Optional: how many segments are heavily padded
    per_segment_padding = padding_mask.mean(axis=1)  # fraction per segment
    print(f"Avg padding per segment: {per_segment_padding.mean():.4f}")
    print(f"% segments >50% padding: {(per_segment_padding > 0.5).mean():.4f}")

    if "test_flow_ids" not in data:
        raise ValueError("data_split.pkl does not contain test_flow_ids.")

    test_flow_ids = data["test_flow_ids"]

    X_test_recon = model.predict(
        X_test,
        batch_size=args.batch_size,
        verbose=1,
    )

    segment_rmse = compute_segment_rmse(X_test, X_test_recon)

    rmse_binary_path = os.path.join(args.experiment_dir, "segment_rmse_distribution_binary.png")
    rmse_subclass_path = os.path.join(args.experiment_dir, "segment_rmse_distribution_subclass.png")

    plot_rmse_histogram_by_class(
        y_test,
        segment_rmse,
        rmse_binary_path,
    )

    plot_rmse_histogram_by_subclass(
        y_test_original,
        segment_rmse,
        rmse_subclass_path,
    )

    print("\nSegment RMSE statistics:")
    for label in sorted(set(y_test_original)):
        mask = y_test_original == label
        scores = segment_rmse[mask]

        print(f"\n{label}")
        print(f"  count : {len(scores)}")
        print(f"  mean  : {np.mean(scores):.6f}")
        print(f"  median: {np.median(scores):.6f}")
        print(f"  std   : {np.std(scores):.6f}")
        print(f"  p90   : {np.percentile(scores, 90):.6f}")
        print(f"  p95   : {np.percentile(scores, 95):.6f}")
        print(f"  p99   : {np.percentile(scores, 99):.6f}")

    roc_path = os.path.join(args.experiment_dir, "segment_roc_curve.png")
    pr_path = os.path.join(args.experiment_dir, "segment_precision_recall_curve.png")

    segment_roc_auc, threshold, best_fpr, best_tpr = plot_roc(
        y_test,
        segment_rmse,
        roc_path,
    )

    segment_pr_auc = plot_precision_recall(
        y_test,
        segment_rmse,
        pr_path,
    )

    segment_pred = (segment_rmse > threshold).astype(int)

    # Binary segment-level confusion matrix
    segment_cm = confusion_matrix(y_test, segment_pred)
    segment_cm_path = os.path.join(args.experiment_dir, "segment_confusion_matrix.png")

    plot_confusion_matrix(
        segment_cm,
        ["Clean", "Malicious"],
        ["Clean", "Malicious"],
        segment_cm_path,
        title="Segment-Level Confusion Matrix",
    )

    # Subclass segment-level confusion matrix
    segment_custom_cm, segment_true_labels, segment_pred_labels = build_custom_confusion_matrix(
        y_test_original,
        segment_pred,
    )

    segment_custom_cm_path = os.path.join(
        args.experiment_dir,
        "segment_custom_confusion_matrix.png",
    )

    plot_confusion_matrix(
        segment_custom_cm,
        segment_pred_labels,
        segment_true_labels,
        segment_custom_cm_path,
        title="Segment-Level Confusion Matrix by True Class",
    )

    # Flow-level aggregation
    flow_results = aggregate_segments_to_flows(
        segment_scores=segment_rmse,
        segment_preds=segment_pred,
        segment_labels=y_test,
        flow_ids=test_flow_ids,
        y_test_original=y_test_original,
        min_anomaly_fraction=args.min_anomaly_fraction,
    )

    y_flow_true, y_flow_pred, y_flow_original = flow_results_to_arrays(flow_results)

    flow_cm = confusion_matrix(y_flow_true, y_flow_pred)
    flow_cm_path = os.path.join(args.experiment_dir, "flow_confusion_matrix_fraction_rule.png")

    plot_confusion_matrix(
        flow_cm,
        ["Clean", "Malicious"],
        ["Clean", "Malicious"],
        flow_cm_path,
        title="Flow-Level Confusion Matrix: Fraction Rule",
    )

    flow_custom_cm, flow_true_labels, flow_pred_labels = build_custom_confusion_matrix(
        y_flow_original,
        y_flow_pred,
    )

    flow_custom_cm_path = os.path.join(
        args.experiment_dir,
        "flow_custom_confusion_matrix_fraction_rule.png",
    )

    plot_confusion_matrix(
        flow_custom_cm,
        flow_pred_labels,
        flow_true_labels,
        flow_custom_cm_path,
        title="Flow-Level Confusion Matrix by True Class",
    )

    segment_report = classification_report(
        y_test,
        segment_pred,
        target_names=["Clean", "Malicious"],
        digits=5,
        output_dict=True,
    )

    flow_report = classification_report(
        y_flow_true,
        y_flow_pred,
        target_names=["Clean", "Malicious"],
        digits=5,
        output_dict=True,
    )

    metrics = {
        "window_size": int(data["window_size"]),
        "threshold_rmse_segment": float(threshold),

        "segment_level": {
            "roc_auc": float(segment_roc_auc),
            "pr_auc": float(segment_pr_auc),
            "best_fpr_from_roc": float(best_fpr),
            "best_tpr_from_roc": float(best_tpr),
            "confusion_matrix": segment_cm.tolist(),
            "custom_confusion_matrix": segment_custom_cm.tolist(),
            "custom_confusion_matrix_true_labels": segment_true_labels,
            "custom_confusion_matrix_pred_labels": segment_pred_labels,
            "classification_report": segment_report,
        },

        "flow_level": {
            "min_anomaly_fraction": float(args.min_anomaly_fraction),
            "confusion_matrix": flow_cm.tolist(),
            "custom_confusion_matrix": flow_custom_cm.tolist(),
            "custom_confusion_matrix_true_labels": flow_true_labels,
            "custom_confusion_matrix_pred_labels": flow_pred_labels,
            "classification_report": flow_report,
        },

        "flow_results": flow_results,
    }

    metrics_path = os.path.join(args.experiment_dir, "time_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Window size:", data["window_size"])
    print("Segment threshold RMSE:", threshold)
    print("Segment ROC-AUC:", segment_roc_auc)
    print("Segment PR-AUC:", segment_pr_auc)

    print("\nSegment-level report:")
    print(classification_report(y_test, segment_pred, target_names=["Clean", "Malicious"], digits=5))

    print("\nFlow-level report: fraction rule")
    print(classification_report(y_flow_true, y_flow_pred, target_names=["Clean", "Malicious"], digits=5))

    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()