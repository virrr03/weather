import os
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "preprocessed_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "rnn_model.h5"
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
    "outputs",
    "rnn_predictions.csv"
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

    df = pd.read_csv(DATA_PATH)
    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH, compile=False)
    data = df[FEATURE_COLS].values.astype(float)
    data = np.clip(data, 0, 1)
    predictions = []

    for i in range(len(data) - TIME_STEPS):
        X = data[i:i+TIME_STEPS]
        X = X.reshape(1, TIME_STEPS, len(FEATURE_COLS))
        pred_scaled = model.predict(X, verbose=0)
        pred_scaled = np.clip(pred_scaled, 0, 1)
        pred = scaler.inverse_transform(pred_scaled)[0]
        predictions.append({
            "index": i + TIME_STEPS,
            "temperature_pred": pred[0],
            "humidity_pred": pred[1],
            "pressure_pred": pred[2],
            "windSpeed_pred": pred[3],
            "irradiance_pred": pred[4]
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    pd.DataFrame(predictions).to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n===================================")
    print("RNN Prediction selesai dibuat")
    print(OUTPUT_PATH)
    print("Jumlah :", len(predictions))
    print("===================================")


if __name__ == "__main__":
    main()