import os
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import firebase_admin

from firebase_admin import credentials, db
from tensorflow.keras.models import load_model

from anfis_interpreter import WeatherANFIS
from evaluate import evaluate_all
from plot_evaluation import plot_all_evaluation


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "rnn_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "data", "models", "scaler.pkl")

SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "config", "weatherprediction.json")

DATABASE_URL = "https://weatherprediction-e48d0-default-rtdb.asia-southeast1.firebasedatabase.app/"

FEATURE_COLS = [
    "temperature",
    "humidity",
    "pressure",
    "windSpeed",
    "rainfall",
    "lightIntensity"
]

TIME_STEPS = 30


if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {
        "databaseURL": DATABASE_URL
    })


def create_sequence(data, time_steps=30):
    X = []

    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])

    return np.array(X)


def clean_prediction(values):
    return {
        "temperature": round(float(values[0]), 2),
        "humidity": round(float(values[1]), 2),
        "pressure": round(float(values[2]), 2),
        "windSpeed": round(max(0, float(values[3])), 2),
        "rainfall": round(max(0, float(values[4])), 2),
        "lightIntensity": round(max(0, float(values[5])), 4)
    }


def predict_multi_steps(model, scaler, data_scaled, n_steps_ahead=24):
    X_input = data_scaled[-TIME_STEPS:].reshape(
        1,
        TIME_STEPS,
        data_scaled.shape[1]
    )

    predictions = []

    for _ in range(n_steps_ahead):
        y_pred_scaled = model.predict(X_input, verbose=0)

        y_pred_original = scaler.inverse_transform(y_pred_scaled)[0]

        prediction = clean_prediction(y_pred_original)

        predictions.append(prediction)

        X_input = np.append(
            X_input[:, 1:, :],
            y_pred_scaled.reshape(1, 1, -1),
            axis=1
        )

    return predictions


def summarize_daily_predictions(predictions, hours_per_day=24):
    daily_summary = []

    total_days = len(predictions) // hours_per_day

    for day in range(total_days):
        start = day * hours_per_day
        end = start + hours_per_day

        day_data = predictions[start:end]

        summary = {
            "day": day + 1,
            "temperature": round(np.mean([p["temperature"] for p in day_data]), 2),
            "humidity": round(np.mean([p["humidity"] for p in day_data]), 2),
            "pressure": round(np.mean([p["pressure"] for p in day_data]), 2),
            "windSpeed": round(np.mean([p["windSpeed"] for p in day_data]), 2),
            "rainfall": round(np.mean([p["rainfall"] for p in day_data]), 2),
            "lightIntensity": round(np.mean([p["lightIntensity"] for p in day_data]), 4)
        }

        daily_summary.append(summary)

    return daily_summary


def get_actual_weather_from_firebase():
    ref = db.reference("sensor_data")
    data = ref.get()

    if not data:
        return None

    latest_key = sorted(data.keys())[-1]
    latest_data = data[latest_key]

    return {
        "temperature": float(latest_data.get("temperature", 0)),
        "humidity": float(latest_data.get("humidity", 0)),
        "pressure": float(latest_data.get("pressure", 0)),
        "windSpeed": float(latest_data.get("windSpeed", 0)),
        "rainfall": float(latest_data.get("rainfall", 0)),
        "lightIntensity": float(latest_data.get("lightIntensity", 0))
    }


def send_to_firebase(data):
    ref = db.reference("weather_predictions")
    ref.push(data)

    print("✅ Semua hasil prediksi berhasil dikirim ke weather_predictions.")


def run_pipeline():
    if not os.path.exists(PROCESSED_FILE):
        print("⚠️ File preprocessed_data.csv belum ada.")
        return

    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model RNN belum ditemukan.")
        return

    if not os.path.exists(SCALER_PATH):
        print("⚠️ Scaler belum ditemukan.")
        return

    df = pd.read_csv(PROCESSED_FILE)
    data_scaled = df[FEATURE_COLS].values

    X = create_sequence(data_scaled, TIME_STEPS)

    if len(X) == 0:
        print("⚠️ Data terlalu sedikit untuk prediksi.")
        return

    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)

    y_pred_scaled = model.predict(X, verbose=0)
    y_actual_scaled = data_scaled[TIME_STEPS:]

    metrics = evaluate_all(
        y_actual_scaled,
        y_pred_scaled,
        FEATURE_COLS
    )

    accuracy_values = []

    for feature in metrics["per_feature"]:
        mape = metrics["per_feature"][feature]["MAPE"]
        accuracy = max(0, 100 - mape)
        accuracy_values.append(accuracy)

    rnn_accuracy = round(np.mean(accuracy_values), 2)

    graph_paths = plot_all_evaluation(metrics)

    actual_weather = get_actual_weather_from_firebase()

    actual_anfis_result = None

    anfis = WeatherANFIS()

    if actual_weather:
        actual_anfis_result = anfis.evaluate(actual_weather)

    predictions_24h = predict_multi_steps(
        model,
        scaler,
        data_scaled,
        n_steps_ahead=24
    )

    prediction_24h_interpretation = []

    for hour, pred in enumerate(predictions_24h, start=1):
        anfis_result = anfis.evaluate(pred)

        prediction_24h_interpretation.append({
            "hour": hour,
            "prediction": pred,
            "anfis": anfis_result
        })

    predictions_7days_hourly = predict_multi_steps(
        model,
        scaler,
        data_scaled,
        n_steps_ahead=168
    )

    predictions_7days = summarize_daily_predictions(
        predictions_7days_hourly
    )

    prediction_data = {
        "timestamp": datetime.now().isoformat(),

        "actual_weather": actual_weather,
        "actual_weather_interpretation": actual_anfis_result,

        "prediction_24h": predictions_24h,
        "prediction_24h_interpretation": prediction_24h_interpretation,

        "prediction_7days": predictions_7days,

        "rnn_accuracy_percent": rnn_accuracy,

        "evaluation": metrics,
        "graphs": graph_paths
    }

    send_to_firebase(prediction_data)

    print("\n=== DATA YANG DIKIRIM KE FIREBASE ===")
    print(prediction_data)


if __name__ == "__main__":
    run_pipeline()