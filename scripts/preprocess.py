# scripts/preprocess.py

import pandas as pd
import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from calibration import apply_sensor_calibration

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_FILE = os.path.join(BASE_DIR, "data", "sensor_data.csv")

PROCESSED_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "preprocessed_data.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "data", "models")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def preprocess():

    if not os.path.exists(RAW_FILE):
        print(f"⚠️ File tidak ditemukan: {RAW_FILE}")
        return

    # BACA DATA =========================
    df = pd.read_csv(RAW_FILE)

    print("=== KOLOM YANG ADA DI RAW FILE ===")
    print(df.columns.tolist())

    print("\n=== DATA AWAL ===")
    print(df.head())

    # IRRADIANCE =========================
    # Firebase memakai nama: irradiance
    # Model RNN juga memakai nama: irradiance

    if "irradiance" not in df.columns:
        print("⚠️ Kolom irradiance tidak ditemukan. irradiance diisi 0.")
        df["irradiance"] = 0
    else:
        df["irradiance"] = pd.to_numeric(
            df["irradiance"],
            errors="coerce"
        ).fillna(0)

    print("\n=== CEK IRRADIANCE SEBELUM DIPILIH ===")
    print(df[["timestamp", "irradiance"]].tail(10))

    # PILIH KOLOM =========================
    selected_cols = [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "windSpeed",
        "irradiance"
    ]

    df = df[selected_cols]

    # TIMESTAMP =========================
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d_%H-%M-%S",
        errors="coerce"
    )

    print("\nJumlah data setelah parsing timestamp:", len(df))
    print(df[["timestamp"]].head())

    df = df.dropna(subset=["timestamp"])

    print("Jumlah data setelah drop timestamp kosong:", len(df))

    # SORTING =========================
    df = df.sort_values("timestamp")

    # HAPUS DUPLIKAT =========================
    df = df.drop_duplicates(subset=["timestamp"])

    # HANDLE MISSING VALUE =========================
    df = df.dropna(how="all")
    df = df.dropna(subset=["timestamp"])

    # Forward fill
    df = df.ffill()

    # UBAH KE NUMERIK =========================
    feature_cols = [
        "temperature",
        "humidity",
        "pressure",
        "windSpeed",
        "irradiance"
    ]

    for col in feature_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(subset=feature_cols)

    print("\n=== DATA SEBELUM KALIBRASI ===")
    print(df[["timestamp"] + feature_cols].tail(10))

    # KALIBRASI SENSOR =========================
    df = apply_sensor_calibration(df)

    print("\n=== DATA SETELAH KALIBRASI SEBELUM NORMALISASI ===")
    print(df[["timestamp"] + feature_cols].tail(10))

    print("\nMax irradiance sebelum normalisasi:", df["irradiance"].max())
    print("Min irradiance sebelum normalisasi:", df["irradiance"].min())

    # NORMALISASI =========================
    scaler = MinMaxScaler()

    df_scaled = df.copy()

    df_scaled[feature_cols] = scaler.fit_transform(
        df[feature_cols]
    )

    print("\n=== DATA SETELAH NORMALISASI ===")
    print(df_scaled[["timestamp"] + feature_cols].tail(10))

    # SIMPAN HASIL =========================
    os.makedirs(
        os.path.dirname(PROCESSED_FILE),
        exist_ok=True
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    df_scaled.to_csv(
        PROCESSED_FILE,
        index=False
    )

    # Simpan scaler
    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print(f"\n✅ Preprocessing selesai")
    print(f"📄 File: {PROCESSED_FILE}")
    print(f"🧠 Scaler: {SCALER_PATH}")

    return df_scaled


if __name__ == "__main__":
    preprocess()