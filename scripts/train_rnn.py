# scripts/train_rnn.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np

#from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "rnn_model.h5")

FEATURE_COLS = [
    "temperature",
    "humidity",
    "pressure",
    "windSpeed",
    "irradiance"
]

TIME_STEPS = 30


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Pastikan semua kolom fitur ada
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ada di data: {missing_cols}")

    # Ambil fitur untuk training
    data = df[FEATURE_COLS].values.astype(float)

    # Karena preprocessed_data.csv kamu sudah dinormalisasi 0-1
    data = np.clip(data, 0, 1)

    print("\n=== DATA TRAINING YANG DIPAKAI ===")
    print(df[FEATURE_COLS].tail(10))
    print(f"\nJumlah data: {len(df)}")
    print(f"Jumlah fitur: {len(FEATURE_COLS)}")
    print(f"Fitur: {FEATURE_COLS}")

    X = data
    y = data

    return X, y


def create_sequences(X, y, seq_length=30):
    Xs, ys = [], []

    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])

    return np.array(Xs), np.array(ys)


def train():
    X_data, y_data = load_data()

    X, y = create_sequences(X_data, y_data, TIME_STEPS)

    if len(X) == 0:
        print("⚠️ Data terlalu sedikit untuk training.")
        print(f"Minimal butuh lebih dari {TIME_STEPS} data.")
        return

    print("\n=== SHAPE DATA TRAINING ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    n = len(X)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    print("\n=== PEMBAGIAN DATA ===")
    print(f"Total Sequence : {n}")
    print(f"Train          : {len(X_train)}")
    print(f"Validation     : {len(X_val)}")
    print(f"Test           : {len(X_test)}")

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        Dense(y.shape[1])
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("\n=== EVALUASI MODEL ===")

    loss, mae = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"Test Loss : {loss:.6f}")
    print(f"Test MAE  : {mae:.6f}")

    print(f"\n✅ Model RNN terbaik berhasil disimpan ke {MODEL_PATH}")


if __name__ == "__main__":
    train()