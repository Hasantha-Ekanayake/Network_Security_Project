#!/usr/bin/env python3

import argparse
import os
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

from dataset_csv import load_dataset

def create_autoencoder(input_dim, latent_dim=4, dropout=0.1):
    input_data = Input(shape=(input_dim,), name="encoder_input")

    x = Dense(latent_dim*8, activation="relu", name="encoder_1")(input_data)
    x = Dropout(dropout)(x)
    x = Dense(latent_dim*4, activation="relu", name="encoder_2")(x)
    x = Dropout(dropout)(x)
    x = Dense(latent_dim*2, activation="relu", name="encoder_3")(x)

    latent = Dense(latent_dim, activation="linear", name="latent_encoding")(x)

    x = Dense(latent_dim*2, activation="relu", name="decoder_1")(latent)
    x = Dense(latent_dim*4, activation="relu", name="decoder_2")(x)
    x = Dropout(dropout)(x)
    x = Dense(latent_dim*8, activation="relu", name="decoder_3")(x)
    x = Dropout(dropout)(x)

    output = Dense(input_dim, activation="linear", name="reconstructed_data")(x)

    model = Model(inputs=input_data, outputs=output, name="doh_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nondoh", default="l1-nondoh.csv")
    parser.add_argument("--benign", default="l2-benign.csv")
    parser.add_argument("--malicious", default="l2-malicious.csv")

    parser.add_argument("--output_dir", default="ae_experiment")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    data = load_dataset(
        nondoh_path=args.nondoh,
        benign_path=args.benign,
        malicious_path=args.malicious,
        random_state=args.random_state
    )

    split_path = os.path.join(args.output_dir, "data_split.pkl")
    with open(split_path, "wb") as f:
        pickle.dump(data, f)

    X_train = data["X_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)

    input_dim = X_train.shape[1]

    model = create_autoencoder(
        input_dim=input_dim,
        latent_dim=args.latent_dim
    )

    print(model.summary())

    current_model_path = os.path.join(args.output_dir, "current_epoch_model.keras")
    best_model_path = os.path.join(args.output_dir, "best_epoch_model.keras")
    log_path = os.path.join(args.output_dir, "training_log.csv")

    callbacks = [
        ModelCheckpoint(
            filepath=current_model_path,
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=False,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            min_delta=1e-5,
            restore_best_weights=True
        ),
        CSVLogger(log_path)
    ]

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks
    )

    history_path = os.path.join(args.output_dir, "history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)

    print("Saved data split to:", split_path)
    print("Saved current model to:", current_model_path)
    print("Saved best model to:", best_model_path)


if __name__ == "__main__":
    main()