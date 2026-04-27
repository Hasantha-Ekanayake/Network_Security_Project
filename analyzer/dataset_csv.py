# dataset_csv.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


STAT_FEATURES_28 = [
    "FlowBytesSent",
    "FlowSentRate",
    "FlowBytesReceived",
    "FlowReceivedRate",

    "PacketLengthMean",
    "PacketLengthMedian",
    "PacketLengthMode",
    "PacketLengthVariance",
    "PacketLengthStandardDeviation",
    "PacketLengthCoefficientofVariation",
    "PacketLengthSkewFromMedian",
    "PacketLengthSkewFromMode",

    "PacketTimeMean",
    "PacketTimeMedian",
    "PacketTimeMode",
    "PacketTimeVariance",
    "PacketTimeStandardDeviation",
    "PacketTimeCoefficientofVariation",
    "PacketTimeSkewFromMedian",
    "PacketTimeSkewFromMode",

    "ResponseTimeTimeMean",
    "ResponseTimeTimeMedian",
    "ResponseTimeTimeMode",
    "ResponseTimeTimeVariance",
    "ResponseTimeTimeStandardDeviation",
    "ResponseTimeTimeCoefficientofVariation",
    "ResponseTimeTimeSkewFromMedian",
    "ResponseTimeTimeSkewFromMode",
]


def load_three_csv_files(nondoh_path, benign_path, malicious_path):
    nondoh_df = pd.read_csv(nondoh_path)
    benign_df = pd.read_csv(benign_path)
    malicious_df = pd.read_csv(malicious_path)

    nondoh_df["Label"] = "NonDoH"
    benign_df["Label"] = "Benign"
    malicious_df["Label"] = "Malicious"

    df = pd.concat([nondoh_df, benign_df, malicious_df], ignore_index=True)

    return df

def clean_numeric_features(
    df,
    feature_cols=STAT_FEATURES_28,
    log_transform=True,
    clip_quantile=0.99,
):
    df = df.copy()

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)

    if log_transform:
        # These statistical features are expected to be nonnegative after clipping.
        df[feature_cols] = df[feature_cols].clip(lower=0)
        df[feature_cols] = np.log1p(df[feature_cols])

    if clip_quantile is not None:
        upper_bounds = df[feature_cols].quantile(clip_quantile)
        df[feature_cols] = df[feature_cols].clip(upper=upper_bounds, axis=1)

    return df

def create_autoencoder_splits(
    df,
    feature_cols=STAT_FEATURES_28,
    label_col="Label",
    clean_labels=("NonDoH", "Benign"),
    anomaly_label="Malicious",
    val_size_clean=0.15,
    test_size_clean=0.15,
    random_state=42,
):
    clean_df = df[df[label_col].isin(clean_labels)].copy()
    anomaly_df = df[df[label_col] == anomaly_label].copy()

    X_clean = clean_df[feature_cols]
    y_clean = clean_df[label_col]

    temp_size = val_size_clean + test_size_clean

    X_train, X_temp, y_train_orig, y_temp_orig = train_test_split(
        X_clean,
        y_clean,
        test_size=temp_size,
        random_state=random_state,
        stratify=y_clean,
    )

    test_ratio = test_size_clean / temp_size

    X_val, X_clean_test, y_val_orig, y_clean_test_orig = train_test_split(
        X_temp,
        y_temp_orig,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y_temp_orig,
    )

    # Match malicious test count to benign test count
    num_malicious_test = len(X_clean_test)

    anomaly_test_df = anomaly_df.sample(
        n=num_malicious_test,
        random_state=random_state,
        replace=False,
    )

    X_anomaly_test = anomaly_test_df[feature_cols]
    y_anomaly_test = anomaly_test_df[label_col]

    X_test = pd.concat([X_clean_test, X_anomaly_test], axis=0)
    y_test_orig = pd.concat([y_clean_test_orig, y_anomaly_test], axis=0)

    y_train = np.zeros(len(X_train), dtype=int)
    y_val = np.zeros(len(X_val), dtype=int)
    y_test = np.where(y_test_orig == anomaly_label, 1, 0)

    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(X_test))

    X_test = X_test.iloc[perm]
    y_test = y_test[perm]
    y_test_orig = y_test_orig.iloc[perm]

    return {
        "X_train": X_train,
        "y_train": y_train,

        "X_val": X_val,
        "y_val": y_val,

        "X_test": X_test,
        "y_test": y_test,
        "y_test_original": y_test_orig.values,

        "feature_cols": feature_cols,
    }


def scale_splits(data):
    scaler = StandardScaler()

    data["X_train"] = scaler.fit_transform(data["X_train"])
    data["X_val"] = scaler.transform(data["X_val"])
    data["X_test"] = scaler.transform(data["X_test"])

    data["scaler"] = scaler

    return data


def load_dataset(
    nondoh_path,
    benign_path,
    malicious_path,
    random_state=42,
    log_transform=True,
    clip_quantile=0.99,
    clean_labels=("NonDoH", "Benign"),
):
    df = load_three_csv_files(nondoh_path, benign_path, malicious_path)

    df = clean_numeric_features(
        df,
        log_transform=log_transform,
        clip_quantile=clip_quantile,
    )

    data = create_autoencoder_splits(
        df=df,
        random_state=random_state,
        clean_labels=clean_labels,
    )

    data = scale_splits(data)

    print("Train:", data["X_train"].shape)
    print("Val:", data["X_val"].shape)
    print("Test:", data["X_test"].shape)
    print("Test clean:", np.sum(data["y_test"] == 0))
    print("Test malicious:", np.sum(data["y_test"] == 1))

    return data