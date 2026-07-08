import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
import pandas as pd
import joblib
import firebase_admin

from firebase_admin import credentials, db
from tensorflow.keras.models import load_model
from datetime import datetime

from anfis_interpreter import WeatherANFIS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "rnn_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "data", "models", "scaler.pkl")

FEATURE_COLS = [
    "temperature",
    "humidity",
    "pressure",
    "windSpeed",
    "irradiance"
]

TIME_STEPS = 30

SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "config", "weatherprediction.json")

DATABASE_URL = "https://weatherprediction-e48d0-default-rtdb.asia-southeast1.firebasedatabase.app/"

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {
        "databaseURL": DATABASE_URL
    })

EVALUATION_METRICS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "results",
    "latest",
    "evaluation_metrics.json"
)

def load_latest_model_accuracy():
    if not os.path.exists(EVALUATION_METRICS_PATH):
        return {
            "model_accuracy_percent": None,
            "accuracy_source": "Belum ada file evaluation_metrics.json",
            "accuracy_per_feature": {}
        }

    try:
        with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        overall_accuracy = metrics.get("overall", {}).get("Accuracy", None)

        per_feature_accuracy = {}

        for feature, values in metrics.get("per_feature", {}).items():
            per_feature_accuracy[feature] = values.get("Accuracy", None)

        return {
            "model_accuracy_percent": overall_accuracy,
            "accuracy_source": "Akurasi diambil dari evaluasi model terakhir",
            "accuracy_per_feature": per_feature_accuracy
        }

    except Exception as e:
        return {
            "model_accuracy_percent": None,
            "accuracy_source": f"Gagal membaca evaluation_metrics.json: {e}",
            "accuracy_per_feature": {}
        }

def load_data():
    df = pd.read_csv(DATA_PATH)

    scaler = joblib.load(SCALER_PATH)

    data_scaled = df[FEATURE_COLS].values.astype(float)

    # karena file preprocessed_data.csv sudah 0-1
    data_scaled = np.clip(data_scaled, 0, 1)

    print("\n=== 10 DATA TERAKHIR CSV SCALED ===")
    print(df[FEATURE_COLS].tail(10))

    return data_scaled, scaler

def predict_multi_steps(n_steps_ahead=24):

    data_scaled, scaler = load_data()

    model = load_model(MODEL_PATH, compile=False)

    X_input = data_scaled[-TIME_STEPS:].reshape(
        1,
        TIME_STEPS,
        data_scaled.shape[1]
    )

    predictions = []

    anfis = WeatherANFIS()

    for hour in range(n_steps_ahead):

        y_pred_scaled = model.predict(X_input, verbose=0)

        y_pred_scaled = np.clip(y_pred_scaled, 0, 1)

        y_pred_original = scaler.inverse_transform(
            y_pred_scaled
        )[0]


        prediction = {
            "hour": hour + 1,
            "temperature": round(max(0, float(y_pred_original[0])), 2),
            "humidity": round(max(0, min(float(y_pred_original[1]), 100)), 2),
            "pressure": round(max(0, float(y_pred_original[2])), 2),
            "windSpeed": round(max(0, float(y_pred_original[3])), 2),
            "irradiance": round(max(0, float(y_pred_original[4])), 2)
        }

        anfis_result = anfis.evaluate(prediction)

        prediction["anfis"] = anfis_result

        predictions.append(prediction)

        X_input = np.append(
            X_input[:, 1:, :],
            y_pred_scaled.reshape(1, 1, -1),
            axis=1
        )

    return predictions

def send_to_firebase(predictions):
    accuracy_info = load_latest_model_accuracy()
    payload = {
        "timestamp": datetime.now().isoformat(),
        "model_accuracy_percent": accuracy_info["model_accuracy_percent"],
        "accuracy_source": accuracy_info["accuracy_source"],
        "accuracy_per_feature": accuracy_info["accuracy_per_feature"],
        "predictions": predictions
    }

    # Data terbaru, selalu ditimpa
    db.reference("predictions_24h").set(payload)

    # Riwayat prediksi, tidak ditimpa
    db.reference("predictions_24h_history").push(payload)

    print("✅ Prediksi 24 jam berhasil dikirim ke Firebase.")
    print(f"✅ Akurasi model terakhir: {accuracy_info['model_accuracy_percent']}%")

if __name__ == "__main__":

    predictions_24h = predict_multi_steps(24)

    send_to_firebase(predictions_24h)

    print("\n=== PREDIKSI 24 JAM ===")

    for p in predictions_24h:

        print(
            f"\nJam +{p['hour']}"
            f"\nSuhu: {p['temperature']} °C"
            f"\nKelembapan: {p['humidity']} %"
            f"\nTekanan: {p['pressure']} hPa"
            f"\nKecepatan Angin: {p['windSpeed']} m/s"
            f"\nIrradiance: {p['irradiance']} lux"
            f"\nRisk: {p['anfis']['risk']}"
            f"\nInterpretasi: {p['anfis']['interpretation']}"
        )
