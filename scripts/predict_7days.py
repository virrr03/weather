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

    # # jaga-jaga kalau file masih punya irradiance tapi belum ada lightIntensity
    # if "lightIntensity" not in df.columns:
    #     if "irradiance" in df.columns:
    #         df["lightIntensity"] = df["irradiance"]
    #     else:
    #         df["lightIntensity"] = 0.0

    scaler = joblib.load(SCALER_PATH)

    data_scaled = df[FEATURE_COLS].values.astype(float)
    data_scaled = np.clip(data_scaled, 0, 1)

    print("\n=== 10 DATA TERAKHIR YANG DIPAKAI MODEL ===")
    print(df[FEATURE_COLS].tail(10))

    return data_scaled, scaler


def predict_multi_steps(n_steps_ahead=168):

    data_scaled, scaler = load_data()

    model = load_model(MODEL_PATH, compile=False)

    X_input = data_scaled[-TIME_STEPS:].reshape(
        1,
        TIME_STEPS,
        data_scaled.shape[1]
    )

    predictions = []

    for _ in range(n_steps_ahead):

        y_pred_scaled = model.predict(X_input, verbose=0)
        y_pred_scaled = np.clip(y_pred_scaled, 0, 1)

        y_pred_original = scaler.inverse_transform(
            y_pred_scaled
        )[0]

        # light_value = round(max(0, float(y_pred_original[5])), 4)

        # # kalau masih bentuk 0-1000, ubah ke 0-1
        # if light_value > 1:
        #     light_value = light_value / 1000.0

        # light_value = round(max(0, min(light_value, 1)), 4)

        prediction = {
            "temperature": round(max(0, float(y_pred_original[0])), 2),
            "humidity": round(max(0, min(float(y_pred_original[1]), 100)), 2),
            "pressure": round(max(0, float(y_pred_original[2])), 2),
            "windSpeed": round(max(0, float(y_pred_original[3])), 2),
            "irradiance": round(max(0, float(y_pred_original[4])), 2)
        }

        predictions.append(prediction)

        X_input = np.append(
            X_input[:, 1:, :],
            y_pred_scaled.reshape(1, 1, -1),
            axis=1
        )

    return predictions


def summarize_daily_predictions(predictions):

    daily_summary = []

    nama_hari = [
        "Senin",
        "Selasa",
        "Rabu",
        "Kamis",
        "Jumat",
        "Sabtu",
        "Minggu"
    ]

    for day in range(7):

        start = day * 24
        end = start + 24

        day_data = predictions[start:end]

        summary = {
            "hari": nama_hari[day],
            "temperature": float(round(np.mean([p["temperature"] for p in day_data]), 2)),
            "humidity": float(round(np.mean([p["humidity"] for p in day_data]), 2)),
            "pressure": float(round(np.mean([p["pressure"] for p in day_data]), 2)),
            "windSpeed": float(round(np.mean([p["windSpeed"] for p in day_data]), 2)),
            "irradiance": float(round(np.mean([p["irradiance"] for p in day_data]), 2))
        }

        daily_summary.append(summary)

    return daily_summary

def make_json_safe(obj):
    """
    Mengubah tipe data numpy menjadi tipe data Python biasa
    agar bisa dikirim ke Firebase.
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)

        if np.isnan(value) or np.isinf(value):
            return None

        return value

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None

    return obj

def send_to_firebase(predictions):
    accuracy_info = load_latest_model_accuracy()

    payload = {
        "timestamp": datetime.now().isoformat(),
        "model_accuracy_percent": accuracy_info["model_accuracy_percent"],
        "accuracy_source": accuracy_info["accuracy_source"],
        "accuracy_per_feature": accuracy_info["accuracy_per_feature"],
        "predictions": predictions
    }

    # Ubah semua tipe numpy menjadi tipe Python biasa
    payload = make_json_safe(payload)

    # Data terbaru, selalu ditimpa
    db.reference("predictions_7days").set(payload)

    # Riwayat prediksi, tidak ditimpa
    db.reference("predictions_7days_history").push(payload)

    print("✅ Prediksi 7 hari berhasil dikirim ke Firebase.")
    print(f"✅ Akurasi model terakhir: {accuracy_info['model_accuracy_percent']}%")

if __name__ == "__main__":

    hourly_predictions = predict_multi_steps(168)

    daily_predictions = summarize_daily_predictions(
        hourly_predictions
    )

    send_to_firebase(daily_predictions)

    print("\n=== PREDIKSI 7 HARI ===")

    for p in daily_predictions:

        print(
            f"\n{p['hari']}"
            f"\nSuhu rata-rata: {p['temperature']} °C"
            f"\nKelembapan rata-rata: {p['humidity']} %"
            f"\nTekanan rata-rata: {p['pressure']} hPa"
            f"\nKecepatan Angin rata-rata: {p['windSpeed']} m/s"
            f"\nIrradiance rata-rata: {p['irradiance']} lux"
        )