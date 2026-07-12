import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RNN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "outputs",
    "rnn_predictions.csv"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "preprocessed_data.csv"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "scaler.pkl"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "anfis_dataset.csv"
)

FEATURE_COLS = [
    "temperature",
    "humidity",
    "pressure",
    "windSpeed",
    "irradiance"
]

TIME_STEPS = 30


def main():

    actual = pd.read_csv(DATA_PATH)

    from joblib import load
    scaler = load(SCALER_PATH)

    actual_original = scaler.inverse_transform(
        actual[FEATURE_COLS].values
    )

    actual = pd.DataFrame(
        actual_original,
        columns=FEATURE_COLS
    )

    pred = pd.read_csv(RNN_PATH)

    actual = actual.iloc[TIME_STEPS:].reset_index(drop=True)

    # Jaga-jaga kalau panjang actual vs pred tidak pas (misal beda 1-3 baris
    # karena proses upstream men-drop row berbeda). Align ke baris terakhir.
    n = min(len(actual), len(pred))
    if len(actual) != len(pred):
        print(f"[WARNING] Panjang actual ({len(actual)}) != pred ({len(pred)}), "
              f"disesuaikan ke {n} baris terakhir.")
        actual = actual.iloc[-n:].reset_index(drop=True)
        pred = pred.iloc[-n:].reset_index(drop=True)

    dataset = pd.DataFrame()

    dataset["temperature_rnn"] = pred["temperature_pred"]
    dataset["humidity_rnn"] = pred["humidity_pred"]
    dataset["pressure_rnn"] = pred["pressure_pred"]
    dataset["windSpeed_rnn"] = pred["windSpeed_pred"]
    dataset["irradiance_rnn"] = pred["irradiance_pred"]

    dataset["temperature_actual"] = actual["temperature"].values
    dataset["humidity_actual"] = actual["humidity"].values
    dataset["pressure_actual"] = actual["pressure"].values
    dataset["windSpeed_actual"] = actual["windSpeed"].values
    dataset["irradiance_actual"] = actual["irradiance"].values

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    dataset.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n===================================")
    print("Dataset ANFIS berhasil dibuat")
    print(OUTPUT_PATH)
    print("Jumlah :", len(dataset))
    print("===================================")


if __name__ == "__main__":
    main()