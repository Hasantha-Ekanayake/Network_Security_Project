#!/usr/bin/env python3

import argparse
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.colors import LogNorm

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)


def compute_rmse(x, x_recon):
    return np.sqrt(np.mean((x - x_recon) ** 2, axis=1))


def plot_roc(y_true, scores, output_path):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

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

    return roc_auc, optimal_threshold, fpr[optimal_idx], tpr[optimal_idx]


def plot_confusion_matrix(cm, output_path):
    plt.figure()
    plt.imshow(cm, norm=LogNorm())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Clean", "Malicious"])
    plt.yticks([0, 1], ["Clean", "Malicious"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def build_custom_confusion_matrix(y_test_original, y_pred):
    true_labels = ["NonDoH", "Benign", "Malicious"]
    pred_labels = ["Clean", "Malicious"]

    cm_custom = np.zeros((len(true_labels), len(pred_labels)), dtype=int)

    for i in range(len(y_test_original)):
        true_label = y_test_original[i]
        pred_label = "Malicious" if y_pred[i] == 1 else "Clean"

        row = true_labels.index(true_label)
        col = pred_labels.index(pred_label)

        cm_custom[row, col] += 1

    return cm_custom, true_labels, pred_labels

def plot_custom_confusion_matrix(cm_custom, true_labels, pred_labels, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_custom, norm=LogNorm())

    plt.title("Custom Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(range(len(pred_labels)), pred_labels)
    plt.yticks(range(len(true_labels)), true_labels)

    for i in range(cm_custom.shape[0]):
        for j in range(cm_custom.shape[1]):
            plt.text(j, i, str(cm_custom[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_dir", default="ae_experiment")
    parser.add_argument("--model_name", default="best_epoch_model.keras")

    args = parser.parse_args()

    split_path = os.path.join(args.experiment_dir, "data_split.pkl")
    model_path = os.path.join(args.experiment_dir, args.model_name)

    with open(split_path, "rb") as f:
        data = pickle.load(f)

    model = tf.keras.models.load_model(model_path)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"]

    y_test_original = data["y_test_original"]

    X_test_recon = model.predict(X_test, batch_size=512, verbose=1)

    rmse_scores = compute_rmse(X_test, X_test_recon)

    roc_path = os.path.join(args.experiment_dir, "roc_curve.png")
    cm_path = os.path.join(args.experiment_dir, "confusion_matrix.png")
    custom_cm_path = os.path.join(args.experiment_dir, "custom_confusion_matrix.png")
    pr_path = os.path.join(args.experiment_dir, "precision_recall_curve.png")

    roc_auc, threshold, best_fpr, best_tpr = plot_roc(
        y_test,
        rmse_scores,
        roc_path
    )

    pr_auc = plot_precision_recall(
        y_test,
        rmse_scores,
        pr_path
    )

    y_pred = (rmse_scores > threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, cm_path)

    custom_cm, true_labels, pred_labels = build_custom_confusion_matrix(
        y_test_original,
        y_pred
    )

    plot_custom_confusion_matrix(
        custom_cm,
        true_labels,
        pred_labels,
        custom_cm_path
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Clean", "Malicious"],
        digits=5,
        output_dict=True
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold_rmse": float(threshold),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_fpr_from_roc": float(best_fpr),
        "best_tpr_from_roc": float(best_tpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)),
        "custom_confusion_matrix": custom_cm.tolist(),
        "custom_confusion_matrix_true_labels": true_labels,
        "custom_confusion_matrix_pred_labels": pred_labels,
        "classification_report": report
    }

    metrics_path = os.path.join(args.experiment_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Threshold RMSE:", threshold)
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    print("Confusion matrix:")
    print(cm)
    print("Custom confusion matrix:")
    print("Rows:", true_labels)
    print("Columns:", pred_labels)
    print(custom_cm)
    print(classification_report(y_test, y_pred, target_names=["Clean", "Malicious"], digits=5))

    print("Saved ROC curve to:", roc_path)
    print("Saved PR curve to:", pr_path)
    print("Saved confusion matrix to:", cm_path)
    print("Saved custom confusion matrix to:", custom_cm_path)
    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()