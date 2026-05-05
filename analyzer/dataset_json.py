# dataset_json.py

import gzip
import json
import math
import numpy as np
import ijson

from sklearn.model_selection import train_test_split


CLUMP_FEATURES_5 = [
    "interarrival",
    "duration",
    "size",
    "pktCount",
    "direction",
]


def normalize(x, data_min, data_max):
    return (x - data_min) / (data_max - data_min)


def load_json_flows(path):
    opener = gzip.open if path.endswith(".gz") else open

    flows = []
    with opener(path, "rb") as f:
        for flow in ijson.items(f, "item"):
            flows.append(flow)

    return flows


def normalize_clump(c):
    """
    Expected input order:
        c[0] = interarrival
        c[1] = duration
        c[2] = size
        c[3] = pktCount
        c[4] = direction
    """

    interarrival = c[0]
    duration = c[1]
    size = c[2]
    pkt_count = c[3]
    direction = c[4]

    return [
        normalize(math.log10(max(1e-12, interarrival)), data_min=-12, data_max=-2),
        normalize(math.log10(max(1e-12, duration)), data_min=-12, data_max=-2),
        normalize(math.log10(min(1e4, max(1.0, size))), data_min=0.5, data_max=4),
        normalize(math.log2(min(256, max(1.0, pkt_count))), data_min=0, data_max=8),
        direction,
    ]


def create_segments_from_flow(flow, window_size):
    clumps = [normalize_clump(c) for c in flow]

    while len(clumps) < window_size:
        clumps.append([-1, -1, -1, -1, 0])

    segments = []

    for start in range(0, len(clumps) - window_size + 1):
        window = clumps[start:start + window_size]
        segments.append(window)

    return segments

def flows_to_segments(flows, window_size, start_flow_id=0):
    segments = []
    flow_ids = []

    for local_id, flow in enumerate(flows):
        global_flow_id = start_flow_id + local_id
        flow_segments = create_segments_from_flow(flow, window_size)

        segments.extend(flow_segments)
        flow_ids.extend([global_flow_id] * len(flow_segments))

    return (
        np.asarray(segments, dtype=np.float32),
        np.asarray(flow_ids, dtype=int)
    )

def load_three_json_files(
    nondoh_path,
    benign_path,
    malicious_path,
    nondoh_ratio=None,
    random_state=42,
):
    rng = np.random.RandomState(random_state)

    nondoh_flows = load_json_flows(nondoh_path)
    benign_flows = load_json_flows(benign_path)
    malicious_flows = load_json_flows(malicious_path)

    if nondoh_ratio is not None:
        max_nondoh = int(nondoh_ratio * len(benign_flows))

        if len(nondoh_flows) > max_nondoh:
            idx = rng.choice(len(nondoh_flows), size=max_nondoh, replace=False)
            nondoh_flows = [nondoh_flows[i] for i in idx]

    all_flows = (
        [(flow, "NonDoH") for flow in nondoh_flows] +
        [(flow, "Benign") for flow in benign_flows] +
        [(flow, "Malicious") for flow in malicious_flows]
    )

    return all_flows


def create_autoencoder_splits(
    all_flows,
    window_size,
    clean_labels=("NonDoH", "Benign"),
    anomaly_label="Malicious",
    val_size_clean=0.15,
    test_size_clean=0.15,
    random_state=42,
):
    rng = np.random.RandomState(random_state)

    clean_flows = [(f, y) for f, y in all_flows if y in clean_labels]
    anomaly_flows = [(f, y) for f, y in all_flows if y == anomaly_label]

    clean_X = [f for f, _ in clean_flows]
    clean_y = [y for _, y in clean_flows]

    anomaly_X = [f for f, _ in anomaly_flows]
    anomaly_y = [y for _, y in anomaly_flows]

    temp_size = val_size_clean + test_size_clean

    stratify_arg = clean_y if len(set(clean_y)) > 1 else None

    X_train_flows, X_temp_flows, y_train_orig, y_temp_orig = train_test_split(
        clean_X,
        clean_y,
        test_size=temp_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_arg,
    )

    test_ratio = test_size_clean / temp_size
    stratify_temp = y_temp_orig if len(set(y_temp_orig)) > 1 else None

    X_val_flows, X_clean_test_flows, y_val_orig, y_clean_test_orig = train_test_split(
        X_temp_flows,
        y_temp_orig,
        test_size=test_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_temp,
    )

    X_train, train_flow_ids = flows_to_segments(X_train_flows, window_size, start_flow_id=0)
    X_val, val_flow_ids = flows_to_segments(X_val_flows, window_size, start_flow_id=0)

    X_clean_test, clean_test_flow_ids = flows_to_segments(
        X_clean_test_flows,
        window_size,
        start_flow_id=0
    )

    X_anomaly_all, anomaly_all_flow_ids = flows_to_segments(
        anomaly_X,
        window_size,
        start_flow_id=len(X_clean_test_flows)
    )

    num_anomaly_test = min(len(X_clean_test), len(X_anomaly_all))
    anomaly_idx = rng.choice(len(X_anomaly_all), size=num_anomaly_test, replace=False)
    X_anomaly_test = X_anomaly_all[anomaly_idx]
    anomaly_test_flow_ids = anomaly_all_flow_ids[anomaly_idx]

    X_test = np.concatenate([X_clean_test, X_anomaly_test], axis=0)
    test_flow_ids = np.concatenate([clean_test_flow_ids, anomaly_test_flow_ids], axis=0)

    y_train = np.zeros(len(X_train), dtype=int)
    y_val = np.zeros(len(X_val), dtype=int)

    y_clean_test = np.zeros(len(X_clean_test), dtype=int)
    y_anomaly_test = np.ones(len(X_anomaly_test), dtype=int)
    y_test = np.concatenate([y_clean_test, y_anomaly_test], axis=0)

    y_clean_test_orig = np.array(["Clean"] * len(X_clean_test), dtype=object)
    y_anomaly_test_orig = np.array([anomaly_label] * len(X_anomaly_test), dtype=object)
    y_test_original = np.concatenate([y_clean_test_orig, y_anomaly_test_orig], axis=0)

    perm = rng.permutation(len(X_test))
    X_test = X_test[perm]
    y_test = y_test[perm]
    test_flow_ids = test_flow_ids[perm]
    y_test_original = y_test_original[perm]

    return {
        "X_train": X_train,
        "y_train": y_train,

        "X_val": X_val,
        "y_val": y_val,

        "X_test": X_test,
        "y_test": y_test,
        "y_test_original": y_test_original,
        "test_flow_ids": test_flow_ids,

        "window_size": window_size,
        "feature_cols": CLUMP_FEATURES_5,
    }


def load_dataset(
    nondoh_path,
    benign_path,
    malicious_path,
    window_size,
    clean_labels=("NonDoH", "Benign"),
    nondoh_ratio=None,
    random_state=42,
):
    all_flows = load_three_json_files(
        nondoh_path=nondoh_path,
        benign_path=benign_path,
        malicious_path=malicious_path,
        nondoh_ratio=nondoh_ratio,
        random_state=random_state,
    )

    data = create_autoencoder_splits(
        all_flows=all_flows,
        window_size=window_size,
        clean_labels=clean_labels,
        random_state=random_state,
    )

    print("Window size:", window_size)
    print("Train:", data["X_train"].shape)
    print("Val:", data["X_val"].shape)
    print("Test:", data["X_test"].shape)
    print("Test clean:", np.sum(data["y_test"] == 0))
    print("Test malicious:", np.sum(data["y_test"] == 1))

    return data