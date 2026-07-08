import os
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db

from anfis_interpreter import WeatherANFIS


BASE_DIR = Path(__file__).resolve().parents[1]

SERVICE_ACCOUNT_PATH = BASE_DIR / "config" / "weatherprediction.json"

DATABASE_URL = "https://weatherprediction-e48d0-default-rtdb.asia-southeast1.firebasedatabase.app/"

STATE_DIR = BASE_DIR / "data" / "runtime"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "actual_anfis_state.json"

SOURCE_PATH = "sensor_data"
OUTPUT_PATH = "actual_weather_interpretations"


if not firebase_admin._apps:
    cred = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))
    firebase_admin.initialize_app(cred, {
        "databaseURL": DATABASE_URL
    })


anfis = WeatherANFIS()


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def normalize_actual_weather(data):
    return {
        "temperature": safe_float(data.get("temperature")),
        "humidity": safe_float(data.get("humidity")),
        "pressure": safe_float(data.get("pressure")),
        "windSpeed": safe_float(data.get("windSpeed")),
        "rainfall": safe_float(data.get("rainfall")),
        "lightIntensity": safe_float(
            data.get("lightIntensity", data.get("irradiance", 0))
        )
    }


def make_hash(data):
    text = json.dumps(data, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_state():
    if not STATE_FILE.exists():
        return {
            "last_key": None,
            "last_hash": None
        }

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "last_key": None,
            "last_hash": None
        }


def save_state(key, data_hash):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "last_key": key,
            "last_hash": data_hash
        }, f, indent=2)


def get_latest_sensor_data():
    ref = db.reference(SOURCE_PATH)

    latest = (
        ref
        .order_by_key()
        .limit_to_last(1)
        .get()
    )

    if not latest:
        return None, None

    latest_key = list(latest.keys())[0]
    latest_data = latest[latest_key]

    if not isinstance(latest_data, dict):
        return None, None

    return latest_key, latest_data


def save_anfis_result(source_key, actual_weather, anfis_result):
    safe_key = (
        source_key
        .replace("/", "_")
        .replace(".", "_")
        .replace("#", "_")
        .replace("$", "_")
        .replace("[", "_")
        .replace("]", "_")
    )

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "source_key": source_key,
        "actual_weather": actual_weather,
        "anfis": anfis_result
    }

    db.reference(OUTPUT_PATH).child(safe_key).set(output_data)

    print(f"✅ ANFIS data aktual berhasil dikirim: {safe_key}")


def process_latest_actual_data():
    latest_key, latest_data = get_latest_sensor_data()

    if latest_key is None or latest_data is None:
        print("⚠️ Belum ada data sensor di Firebase.")
        return

    actual_weather = normalize_actual_weather(latest_data)
    current_hash = make_hash(actual_weather)

    state = load_state()

    same_key = latest_key == state.get("last_key")
    same_data = current_hash == state.get("last_hash")

    if same_key and same_data:
        print("ℹ️ Data aktual belum berubah, tidak diproses ulang.")
        return

    anfis_result = anfis.evaluate(actual_weather)

    save_anfis_result(
        source_key=latest_key,
        actual_weather=actual_weather,
        anfis_result=anfis_result
    )

    save_state(latest_key, current_hash)

    print("=== DATA AKTUAL ===")
    print(actual_weather)

    print("=== HASIL ANFIS ===")
    print(anfis_result)


def firebase_callback(event):
    try:
        print(f"\n📡 Perubahan Firebase terdeteksi: {event.path}")
        process_latest_actual_data()

    except Exception as e:
        print(f"❌ Error saat proses ANFIS aktual: {e}")


def main():
    print("🚀 Actual ANFIS listener berjalan...")
    print(f"📌 Membaca perubahan dari Firebase path: {SOURCE_PATH}")
    print(f"📌 Menyimpan hasil ke Firebase path: {OUTPUT_PATH}")

    ref = db.reference(SOURCE_PATH)
    listener = ref.listen(firebase_callback)

    try:
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("\nListener dihentikan.")
        listener.close()


if __name__ == "__main__":
    main()