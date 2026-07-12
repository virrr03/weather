# scripts/run_evaluation.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

from evaluate import evaluate_all_original_scale
from plot_evaluation import plot_all_evaluation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "rnn_model.h5")

# Scaler yang dipakai saat preprocessing RNN (fit di data training).
# SESUAIKAN nama file ini dengan yang benar-benar dipakai train_rnn.py --
# harus scaler yang SAMA persis, bukan yang di-fit ulang di sini.
SCALER_PATH = os.path.join(BASE_DIR, "data", "models", "scaler.pkl")

RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
LATEST_DIR = os.path.join(RESULTS_DIR, "latest")
HISTORY_DIR = os.path.join(RESULTS_DIR, "history")

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


def save_to_history(metrics, n_total, n_test):
    """
    Menyimpan hasil evaluasi ke folder history dengan timestamp,
    supaya setiap kali evaluasi dijalankan ulang (misalnya setelah
    perbaikan pipeline seperti filter outlier), hasil sebelumnya
    tidak tertimpa dan tetap bisa dibandingkan/dilampirkan di skripsi.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_sequence": n_total,
        "test_sequence": n_test,
        "metrics": metrics,
    }

    history_path = os.path.join(HISTORY_DIR, f"evaluation_{timestamp}.json")
    with open(history_path, "w") as f:
        json.dump(record, f, indent=4)

    print(f"📁 Hasil evaluasi disimpan ke history: {history_path}")

    # Simpan juga versi ringkas dalam bentuk tabel CSV yang terus
    # bertambah (append), supaya mudah dibandingkan antar percobaan
    # dalam satu file untuk dilampirkan di skripsi.
    summary_row = {
        "timestamp": record["timestamp"],
        "total_sequence": n_total,
        "test_sequence": n_test,
    }
    for feature, m in metrics["per_feature"].items():
        summary_row[f"{feature}_MAE"] = m["MAE"]
        summary_row[f"{feature}_RMSE"] = m["RMSE"]
        summary_row[f"{feature}_MAPE"] = m["MAPE"]
        summary_row[f"{feature}_R2"] = m["R2_Score"]
    summary_row["overall_MAE"] = metrics["overall"]["MAE"]
    summary_row["overall_RMSE"] = metrics["overall"]["RMSE"]
    summary_row["overall_MAPE"] = metrics["overall"]["MAPE"]
    summary_row["overall_R2"] = metrics["overall"]["R2_Score"]

    summary_path = os.path.join(HISTORY_DIR, "evaluation_summary.csv")

    df_row = pd.DataFrame([summary_row])
    if os.path.exists(summary_path):
        df_row.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(summary_path, mode="w", header=True, index=False)

    print(f"📊 Ringkasan ditambahkan ke: {summary_path}")

    return history_path, summary_path


def main():
    print("\n=== EVALUASI MODEL RNN DIMULAI (TEST SET MURNI) ===")

    df = pd.read_csv(DATA_PATH)

    # data_scaled di sini TETAP hasil scaler.transform() (0-1) -- itu memang
    # yang dibutuhkan sebagai input model. clip hanya jaga-jaga floating point,
    # BUKAN untuk "membuang" rentang datanya.
    data_scaled = df[FEATURE_COLS].values.astype(float)
    data_scaled = np.clip(data_scaled, 0, 1)

    X, y_actual_scaled = create_sequences(data_scaled, TIME_STEPS)

    if len(X) == 0:
        print("⚠️ Data terlalu sedikit untuk evaluasi.")
        return

    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_test = X[val_end:]
    y_test_scaled = y_actual_scaled[val_end:]

    print(f"Total sequence : {n}")
    print(f"Test (murni)   : {len(X_test)}  (index {val_end}..{n-1})")

    model = load_model(MODEL_PATH, compile=False)

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_scaled = np.clip(y_pred_scaled, 0, 1)

    # Muat scaler yang SAMA dipakai saat fit preprocessing RNN, urutan
    # kolom harus sama dengan FEATURE_COLS.
    y_scaler = joblib.load(SCALER_PATH)

    # evaluate_all_original_scale() melakukan inverse_transform sendiri,
    # sehingga metrik dihitung di satuan asli (°C, %, hPa, m/s, W/m²),
    # bukan di ruang ternormalisasi 0-1 -- supaya apple-to-apple dengan
    # kolom RNN_* di rnn_vs_anfis_comparison.csv.
    metrics = evaluate_all_original_scale(
        y_test_scaled, y_pred_scaled, FEATURE_COLS, y_scaler
    )

    saved_files = plot_all_evaluation(metrics)

    history_path, summary_path = save_to_history(metrics, n, len(X_test))

    print("\n=== HASIL EVALUASI (TEST SET MURNI, SATUAN ASLI) ===")
    print(json.dumps(metrics, indent=4))

    print("\n=== FILE EVALUASI TERSIMPAN ===")
    print(json.dumps(saved_files, indent=4))
    print("History JSON  :", history_path)
    print("History CSV   :", summary_path)

    print("\n✅ Evaluasi model (test murni, satuan asli, tanpa leakage) berhasil disimpan.")
if __name__ == "__main__":
    main()