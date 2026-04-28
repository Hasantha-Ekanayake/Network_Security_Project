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
    average_precision_score
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
    plt.title("ROC Curve")
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
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return pr_auc


def plot_confusion_matrix(cm, output_path, title):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Clean", "Malicious"])
    plt.yticks([0, 1], ["Clean", "Malicious"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            plt.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=color)

    plt.colorbar(label="Row-normalized percentage")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def aggregate_segments_to_flows(
    segment_scores,
    segment_preds,
    segment_labels,
    flow_ids,
    y_test_original,
    min_anomaly_fraction=0.30,
    min_consecutive_anomalies=3,
):
    flow_results = {}

    for fid in np.unique(flow_ids):
        mask = flow_ids == fid

        scores = segment_scores[mask]
        preds = segment_preds[mask]
        labels = segment_labels[mask]
        orig_labels = y_test_original[mask]

        true_label = int(np.max(labels))

        anomaly_fraction = float(np.mean(preds))
        max_score = float(np.max(scores))
        mean_score = float(np.mean(scores))

        max_consecutive = 0
        current = 0
        for p in preds:
            if p == 1:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0

        flow_pred_fraction = int(anomaly_fraction >= min_anomaly_fraction)
        flow_pred_consecutive = int(max_consecutive >= min_consecutive_anomalies)
        flow_pred_combined = int(
            flow_pred_fraction == 1 or flow_pred_consecutive == 1
        )

        flow_results[int(fid)] = {
            "true_label": true_label,
            "true_original": str(orig_labels[0]),
            "num_segments": int(len(scores)),
            "max_score": max_score,
            "mean_score": mean_score,
            "anomaly_fraction": anomaly_fraction,
            "max_consecutive_anomalies": int(max_consecutive),
            "pred_fraction_rule": flow_pred_fraction,
            "pred_consecutive_rule": flow_pred_consecutive,
            "pred_combined_rule": flow_pred_combined,
        }

    return flow_results


def flow_results_to_arrays(flow_results, pred_key):
    y_true = []
    y_pred = []

    for _, r in flow_results.items():
        y_true.append(r["true_label"])
        y_pred.append(r[pred_key])

    return np.asarray(y_true), np.asarray(y_pred)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--model_name", default="best_epoch_model.keras")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--min_anomaly_fraction", type=float, default=0.30)
    parser.add_argument("--min_consecutive_anomalies", type=int, default=3)

    args = parser.parse_args()

    split_path = os.path.join(args.experiment_dir, "data_split.pkl")
    model_path = os.path.join(args.experiment_dir, args.model_name)

    with open(split_path, "rb") as f:
        data = pickle.load(f)

    model = tf.keras.models.load_model(model_path)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"]
    y_test_original = data["y_test_original"]

    if "test_flow_ids" not in data:
        raise ValueError(
            "data_split.pkl does not contain test_flow_ids. "
            "Update dataset_json.py to preserve flow IDs before testing."
        )

    test_flow_ids = data["test_flow_ids"]

    X_test_recon = model.predict(
        X_test,
        batch_size=args.batch_size,
        verbose=1
    )

    segment_rmse = compute_segment_rmse(X_test, X_test_recon)

    roc_path = os.path.join(args.experiment_dir, "segment_roc_curve.png")
    pr_path = os.path.join(args.experiment_dir, "segment_precision_recall_curve.png")

    segment_roc_auc, threshold, best_fpr, best_tpr = plot_roc(
        y_test,
        segment_rmse,
        roc_path
    )

    segment_pr_auc = plot_precision_recall(
        y_test,
        segment_rmse,
        pr_path
    )

    segment_pred = (segment_rmse > threshold).astype(int)

    segment_cm = confusion_matrix(y_test, segment_pred)
    segment_cm_path = os.path.join(args.experiment_dir, "segment_confusion_matrix.png")
    plot_confusion_matrix(
        segment_cm,
        segment_cm_path,
        title="Segment-Level Confusion Matrix"
    )

    flow_results = aggregate_segments_to_flows(
        segment_scores=segment_rmse,
        segment_preds=segment_pred,
        segment_labels=y_test,
        flow_ids=test_flow_ids,
        y_test_original=y_test_original,
        min_anomaly_fraction=args.min_anomaly_fraction,
        min_consecutive_anomalies=args.min_consecutive_anomalies,
    )

    y_flow_true, y_flow_pred_fraction = flow_results_to_arrays(
        flow_results,
        "pred_fraction_rule"
    )

    _, y_flow_pred_consecutive = flow_results_to_arrays(
        flow_results,
        "pred_consecutive_rule"
    )

    _, y_flow_pred_combined = flow_results_to_arrays(
        flow_results,
        "pred_combined_rule"
    )

    flow_cm_fraction = confusion_matrix(y_flow_true, y_flow_pred_fraction)
    flow_cm_consecutive = confusion_matrix(y_flow_true, y_flow_pred_consecutive)
    flow_cm_combined = confusion_matrix(y_flow_true, y_flow_pred_combined)

    plot_confusion_matrix(
        flow_cm_fraction,
        os.path.join(args.experiment_dir, "flow_confusion_matrix_fraction_rule.png"),
        title="Flow-Level Confusion Matrix: Fraction Rule"
    )

    plot_confusion_matrix(
        flow_cm_consecutive,
        os.path.join(args.experiment_dir, "flow_confusion_matrix_consecutive_rule.png"),
        title="Flow-Level Confusion Matrix: Consecutive Rule"
    )

    plot_confusion_matrix(
        flow_cm_combined,
        os.path.join(args.experiment_dir, "flow_confusion_matrix_combined_rule.png"),
        title="Flow-Level Confusion Matrix: Combined Rule"
    )

    segment_report = classification_report(
        y_test,
        segment_pred,
        target_names=["Clean", "Malicious"],
        digits=5,
        output_dict=True
    )

    flow_report_fraction = classification_report(
        y_flow_true,
        y_flow_pred_fraction,
        target_names=["Clean", "Malicious"],
        digits=5,
        output_dict=True
    )

    flow_report_consecutive = classification_report(
        y_flow_true,
        y_flow_pred_consecutive,
        target_names=["Clean", "Malicious"],
        digits=5,
        output_dict=True
    )

    flow_report_combined = classification_report(
        y_flow_true,
        y_flow_pred_combined,
        target_names=["Clean", "Malicious"],
        digits=5,
        output_dict=True
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
            "classification_report": segment_report,
        },

        "flow_level": {
            "min_anomaly_fraction": float(args.min_anomaly_fraction),
            "min_consecutive_anomalies": int(args.min_consecutive_anomalies),

            "fraction_rule": {
                "confusion_matrix": flow_cm_fraction.tolist(),
                "classification_report": flow_report_fraction,
            },

            "consecutive_rule": {
                "confusion_matrix": flow_cm_consecutive.tolist(),
                "classification_report": flow_report_consecutive,
            },

            "combined_rule": {
                "confusion_matrix": flow_cm_combined.tolist(),
                "classification_report": flow_report_combined,
            },
        },

        "flow_results": flow_results,
    }

    metrics_path = os.path.join(args.experiment_dir, "time_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Segment threshold RMSE:", threshold)
    print("Segment ROC-AUC:", segment_roc_auc)
    print("Segment PR-AUC:", segment_pr_auc)
    print("\nSegment-level report:")
    print(classification_report(y_test, segment_pred, target_names=["Clean", "Malicious"], digits=5))

    print("\nFlow-level report: fraction rule")
    print(classification_report(y_flow_true, y_flow_pred_fraction, target_names=["Clean", "Malicious"], digits=5))

    print("\nFlow-level report: consecutive rule")
    print(classification_report(y_flow_true, y_flow_pred_consecutive, target_names=["Clean", "Malicious"], digits=5))

    print("\nFlow-level report: combined rule")
    print(classification_report(y_flow_true, y_flow_pred_combined, target_names=["Clean", "Malicious"], digits=5))

    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()