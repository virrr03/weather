import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "processed", "anfis_dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
SPLIT_PATH = os.path.join(BASE_DIR, "data", "processed", "anfis_split.npz")

FITUR_INPUT = [
    "temperature_rnn",
    "humidity_rnn",
    "pressure_rnn",
    "windSpeed_rnn",
    "irradiance_rnn",
]

FITUR_OUTPUT = [
    "temperature_actual",
    "humidity_actual",
    "pressure_actual",
    "windSpeed_actual",
    "irradiance_actual",
]

NAMA_VARIABEL = [
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "Irradiance",
]


def main():
    print("\n===================================")
    print("PREPROCESSING DATASET ANFIS")
    print("===================================")

    df = pd.read_csv(DATASET_PATH)

    X = df[FITUR_INPUT].values
    y = df[FITUR_OUTPUT].values

    print("Jumlah Data :", len(df))
    print("Input Shape :", X.shape)
    print("Target Shape:", y.shape)

    # ============================
    # SPLIT DATA (time-series, shuffle=False)
    # ============================
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, shuffle=False
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, shuffle=False
    )

    print("\nTrain :", len(X_train))
    print("Valid :", len(X_valid))
    print("Test  :", len(X_test))

    # Simpan versi ASLI (belum dinormalisasi) untuk X_valid/X_test.
    # Ini dipakai nanti di tahap evaluasi sebagai baseline "RNN saja"
    # (karena X = hasil prediksi RNN), dibandingkan dengan "RNN + ANFIS".
    X_valid_orig = X_valid.copy()
    X_test_orig = X_test.copy()
    y_valid_orig = y_valid.copy()
    y_test_orig = y_test.copy()

    # ============================
    # NORMALISASI
    # ============================
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_valid_scaled = x_scaler.transform(X_valid)
    X_test_scaled = x_scaler.transform(X_test)

    y_train_scaled = y_scaler.fit_transform(y_train)
    y_valid_scaled = y_scaler.transform(y_valid)
    y_test_scaled = y_scaler.transform(y_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)

    joblib.dump(x_scaler, os.path.join(MODEL_DIR, "anfis_x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(MODEL_DIR, "anfis_y_scaler.pkl"))

    # ============================
    # SIMPAN HASIL SPLIT
    # ============================
    np.savez(
        SPLIT_PATH,
        X_train_scaled=X_train_scaled,
        X_valid_scaled=X_valid_scaled,
        X_test_scaled=X_test_scaled,
        y_train_scaled=y_train_scaled,
        y_valid_scaled=y_valid_scaled,
        y_test_scaled=y_test_scaled,
        X_valid_orig=X_valid_orig,
        X_test_orig=X_test_orig,
        y_valid_orig=y_valid_orig,
        y_test_orig=y_test_orig,
    )

    print("\n===================================")
    print("PREPROCESSING SELESAI")
    print("===================================")
    print("Data split & scaler disimpan di:")
    print(" -", SPLIT_PATH)
    print(" -", os.path.join(MODEL_DIR, "anfis_x_scaler.pkl"))
    print(" -", os.path.join(MODEL_DIR, "anfis_y_scaler.pkl"))


if __name__ == "__main__":
    main()