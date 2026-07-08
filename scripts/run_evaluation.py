import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from evaluate import evaluate_all
from plot_evaluation import plot_all_evaluation

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


def create_sequences(data, seq_length):
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


def main():
    print("\n=== EVALUASI MODEL RNN DIMULAI ===")

    df = pd.read_csv(DATA_PATH)

    data_scaled = df[FEATURE_COLS].values.astype(float)
    data_scaled = np.clip(data_scaled, 0, 1)

    X, y_actual = create_sequences(data_scaled, TIME_STEPS)

    if len(X) == 0:
        print("⚠️ Data terlalu sedikit untuk evaluasi.")
        return

    model = load_model(MODEL_PATH, compile=False)

    y_pred = model.predict(X, verbose=0)
    y_pred = np.clip(y_pred, 0, 1)

    metrics = evaluate_all(y_actual, y_pred, FEATURE_COLS)

    saved_files = plot_all_evaluation(metrics)

    print("\n=== HASIL EVALUASI ===")
    import json
    print(json.dumps(metrics, indent=4))

    print("\n=== FILE EVALUASI TERSIMPAN ===")
    print(json.dumps(saved_files, indent=4))

    print("\n✅ Evaluasi model dan plot berhasil disimpan ke latest dan history.")


if __name__ == "__main__":
    main()