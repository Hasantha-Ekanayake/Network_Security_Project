# dataset_json.py

import gzip
import math
import pickle
import os
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


def normalize_flow(flow):
    return [normalize_clump(c) for c in flow]


def create_segments_from_normalized_flow(norm_flow, min_window_size, max_window_size):
    clumps = list(norm_flow)
    original_len = len(clumps)

    # Pad enough so the smallest window can slide to the end,
    # while each stored segment still has max_window_size length.
    padded_len = max(
        original_len + max_window_size - min_window_size,
        max_window_size,
    )

    while len(clumps) < padded_len:
        clumps.append([-1, -1, -1, -1, 0])

    num_starts = max(1, original_len - min_window_size + 1)

    segments = []
    for start in range(num_starts):
        segments.append(clumps[start:start + max_window_size])

    return segments


def flows_to_max_segments(
    normalized_flows,
    min_window_size,
    max_window_size,
    start_flow_id=0,
    flow_labels=None,
):
    segments = []
    flow_ids = []
    segment_labels = []

    for local_id, flow in enumerate(normalized_flows):
        global_flow_id = start_flow_id + local_id

        flow_segments = create_segments_from_normalized_flow(
            flow,
            min_window_size=min_window_size,
            max_window_size=max_window_size,
        )

        segments.extend(flow_segments)
        flow_ids.extend([global_flow_id] * len(flow_segments))

        if flow_labels is not None:
            segment_labels.extend([flow_labels[local_id]] * len(flow_segments))

    outputs = [
        np.asarray(segments, dtype=np.float32),
        np.asarray(flow_ids, dtype=int),
    ]

    if flow_labels is not None:
        outputs.append(np.asarray(segment_labels, dtype=object))

    return tuple(outputs)


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


def create_flow_splits(
    all_flows,
    clean_labels=("NonDoH", "Benign"),
    anomaly_label="Malicious",
    val_size_clean=0.15,
    test_size_clean=0.15,
    random_state=42,
):
    clean_flows = [(f, y) for f, y in all_flows if y in clean_labels]
    anomaly_flows = [(f, y) for f, y in all_flows if y == anomaly_label]

    clean_X = [f for f, _ in clean_flows]
    clean_y = [y for _, y in clean_flows]

    anomaly_X = [f for f, _ in anomaly_flows]

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

    return {
        "train_flows": X_train_flows,
        "val_flows": X_val_flows,
        "clean_test_flows": X_clean_test_flows,
        "clean_test_labels": y_clean_test_orig,
        "anomaly_flows": anomaly_X,
    }


def normalize_flow_splits(flow_splits):
    return {
        "train_flows": [normalize_flow(f) for f in flow_splits["train_flows"]],
        "val_flows": [normalize_flow(f) for f in flow_splits["val_flows"]],
        "clean_test_flows": [normalize_flow(f) for f in flow_splits["clean_test_flows"]],
        "clean_test_labels": flow_splits["clean_test_labels"],
        "anomaly_flows": [normalize_flow(f) for f in flow_splits["anomaly_flows"]],
    }


def create_max_window_dataset(
    normalized_splits,
    min_window_size,
    max_window_size,
    anomaly_label="Malicious",
    random_state=42,
):
    rng = np.random.RandomState(random_state)

    X_train_max, train_flow_ids = flows_to_max_segments(
        normalized_splits["train_flows"],
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        start_flow_id=0,
    )

    X_val_max, val_flow_ids = flows_to_max_segments(
        normalized_splits["val_flows"],
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        start_flow_id=0,
    )

    X_clean_test_max, clean_test_flow_ids, clean_test_segment_labels = flows_to_max_segments(
        normalized_splits["clean_test_flows"],
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        start_flow_id=0,
        flow_labels=normalized_splits["clean_test_labels"],
    )

    X_anomaly_all_max, anomaly_all_flow_ids = flows_to_max_segments(
        normalized_splits["anomaly_flows"],
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        start_flow_id=len(normalized_splits["clean_test_flows"]),
    )

    num_anomaly_test = min(len(X_clean_test_max), len(X_anomaly_all_max))

    anomaly_idx = rng.choice(
        len(X_anomaly_all_max),
        size=num_anomaly_test,
        replace=False,
    )

    X_anomaly_test_max = X_anomaly_all_max[anomaly_idx]
    anomaly_test_flow_ids = anomaly_all_flow_ids[anomaly_idx]

    X_test_max = np.concatenate([X_clean_test_max, X_anomaly_test_max], axis=0)

    y_train = np.zeros(len(X_train_max), dtype=int)
    y_val = np.zeros(len(X_val_max), dtype=int)

    y_clean_test = np.zeros(len(X_clean_test_max), dtype=int)
    y_anomaly_test = np.ones(len(X_anomaly_test_max), dtype=int)
    y_test = np.concatenate([y_clean_test, y_anomaly_test], axis=0)

    # Now preserves clean subclass labels: NonDoH / Benign / Malicious
    y_clean_test_orig = clean_test_segment_labels
    y_anomaly_test_orig = np.array([anomaly_label] * len(X_anomaly_test_max), dtype=object)
    y_test_original = np.concatenate([y_clean_test_orig, y_anomaly_test_orig], axis=0)

    test_flow_ids = np.concatenate(
        [clean_test_flow_ids, anomaly_test_flow_ids],
        axis=0,
    )

    perm = rng.permutation(len(X_test_max))

    X_test_max = X_test_max[perm]
    y_test = y_test[perm]
    y_test_original = y_test_original[perm]
    test_flow_ids = test_flow_ids[perm]

    return {
        "X_train_max": X_train_max,
        "y_train": y_train,

        "X_val_max": X_val_max,
        "y_val": y_val,

        "X_test_max": X_test_max,
        "y_test": y_test,
        "y_test_original": y_test_original,
        "test_flow_ids": test_flow_ids,

        "max_window_size": max_window_size,
        "min_window_size": min_window_size,
        "feature_cols": CLUMP_FEATURES_5,
    }


def load_max_window_dataset(
    nondoh_path,
    benign_path,
    malicious_path,
    min_window_size,
    max_window_size,
    clean_labels=("NonDoH", "Benign"),
    nondoh_ratio=None,
    random_state=42,
    cache_path=None,
):
    if cache_path is not None and os.path.exists(cache_path):
        print("Using cached max-window dataset:", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    all_flows = load_three_json_files(
        nondoh_path=nondoh_path,
        benign_path=benign_path,
        malicious_path=malicious_path,
        nondoh_ratio=nondoh_ratio,
        random_state=random_state,
    )

    flow_splits = create_flow_splits(
        all_flows=all_flows,
        clean_labels=clean_labels,
        random_state=random_state,
    )

    normalized_splits = normalize_flow_splits(flow_splits)

    max_data = create_max_window_dataset(
        normalized_splits=normalized_splits,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        random_state=random_state,
    )

    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(max_data, f)

    print("Min window size:", min_window_size)
    print("Max window size:", max_window_size)
    print("Train max:", max_data["X_train_max"].shape)
    print("Val max:", max_data["X_val_max"].shape)
    print("Test max:", max_data["X_test_max"].shape)
    print("Test clean:", np.sum(max_data["y_test"] == 0))
    print("Test malicious:", np.sum(max_data["y_test"] == 1))

    unique, counts = np.unique(max_data["y_test_original"], return_counts=True)
    print("Test original label counts:", dict(zip(unique, counts)))

    return max_data


def load_dataset(
    nondoh_path,
    benign_path,
    malicious_path,
    min_window_size,
    max_window_size,
    clean_labels=("NonDoH", "Benign"),
    nondoh_ratio=None,
    random_state=42,
):
    max_data = load_max_window_dataset(
        nondoh_path=nondoh_path,
        benign_path=benign_path,
        malicious_path=malicious_path,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        clean_labels=clean_labels,
        nondoh_ratio=nondoh_ratio,
        random_state=random_state,
    )

    return max_data