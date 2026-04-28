#!/usr/bin/env python3

import argparse
import os
import pickle
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

from dataset_json import load_dataset


def create_lstm_autoencoder(window_size, num_features=5, latent_dim=16, lstm_units=64, dropout=0.0):
    input_seq = Input(shape=(window_size, num_features), name="input_sequence")

    x = LSTM(lstm_units, return_sequences=True, name="encoder_lstm_1")(input_seq)
    if dropout > 0:
        x = Dropout(dropout)(x)

    x = LSTM(lstm_units // 2, return_sequences=False, name="encoder_lstm_2")(x)

    latent = Dense(latent_dim, activation="linear", name="latent_encoding")(x)

    x = RepeatVector(window_size, name="repeat_latent")(latent)

    x = LSTM(lstm_units // 2, return_sequences=True, name="decoder_lstm_1")(x)
    if dropout > 0:
        x = Dropout(dropout)(x)

    x = LSTM(lstm_units, return_sequences=True, name="decoder_lstm_2")(x)

    output_seq = TimeDistributed(
        Dense(num_features, activation="linear"),
        name="reconstructed_sequence"
    )(x)

    model = Model(inputs=input_seq, outputs=output_seq, name="doh_lstm_autoencoder")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    return model


def train_one_window_size(args, window_size):
    print("=" * 80)
    print(f"Training LSTM autoencoder for window_size = {window_size}")
    print("=" * 80)

    output_dir = os.path.join(args.output_dir, f"window_{window_size}")
    os.makedirs(output_dir, exist_ok=True)

    data = load_dataset(
        nondoh_path=args.nondoh,
        benign_path=args.benign,
        malicious_path=args.malicious,
        window_size=window_size,
        clean_labels=tuple(args.clean_labels),
        nondoh_ratio=args.nondoh_ratio,
        random_state=args.random_state,
    )

    split_path = os.path.join(output_dir, "data_split.pkl")
    with open(split_path, "wb") as f:
        pickle.dump(data, f)

    X_train = data["X_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)

    model = create_lstm_autoencoder(
        window_size=window_size,
        num_features=X_train.shape[2],
        latent_dim=args.latent_dim,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
    )

    print(model.summary())

    current_model_path = os.path.join(output_dir, "current_epoch_model.keras")
    best_model_path = os.path.join(output_dir, "best_epoch_model.keras")
    log_path = os.path.join(output_dir, "training_log.csv")

    callbacks = [
        ModelCheckpoint(
            filepath=current_model_path,
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=False,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            min_delta=1e-5,
            restore_best_weights=True,
        ),
        CSVLogger(log_path),
    ]

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
    )

    history_path = os.path.join(output_dir, "history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)

    best_val_loss = float(np.min(history.history["val_loss"]))

    summary = {
        "window_size": window_size,
        "best_val_loss": best_val_loss,
        "best_val_rmse": float(np.sqrt(best_val_loss)),
        "epochs_trained": len(history.history["loss"]),
        "output_dir": output_dir,
    }

    summary_path = os.path.join(output_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Best val loss for window {window_size}: {best_val_loss}")
    print(f"Best val RMSE for window {window_size}: {np.sqrt(best_val_loss)}")

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nondoh", default="data/final/ndoh.json")
    parser.add_argument("--benign", default="data/final/benign_doh.json")
    parser.add_argument("--malicious", default="data/final/malicious_doh.json")

    parser.add_argument("--output_dir", default="lstm_ae_experiment")

    parser.add_argument("--window_min", type=int, default=4)
    parser.add_argument("--window_max", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--lstm_units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument(
        "--clean_labels",
        nargs="+",
        default=["Benign"],
        help="Labels to treat as clean, e.g., Benign or NonDoH Benign",
    )

    parser.add_argument(
        "--nondoh_ratio",
        type=float,
        default=None,
        help="Limit NonDoH to ratio * Benign flow count",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    all_summaries = []

    for window_size in range(args.window_min, args.window_max + 1):
        summary = train_one_window_size(args, window_size)
        all_summaries.append(summary)

    all_summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    with open(all_summary_path, "w") as f:
        json.dump(all_summaries, f, indent=4)

    best = min(all_summaries, key=lambda x: x["best_val_loss"])

    print("\n" + "=" * 80)
    print("Ablation complete")
    print("=" * 80)
    print("Best window size:", best["window_size"])
    print("Best val loss:", best["best_val_loss"])
    print("Best val RMSE:", best["best_val_rmse"])
    print("Best model directory:", best["output_dir"])
    print("Saved ablation summary to:", all_summary_path)


if __name__ == "__main__":
    main()