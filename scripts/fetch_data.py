import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "weatherprediction.json")

DATABASE_URL = "https://weatherprediction-e48d0-default-rtdb.asia-southeast1.firebasedatabase.app/"

DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_FILE = os.path.join(DATA_DIR, "sensor_data.csv")

if not firebase_admin._apps:
    cred = credentials.Certificate(CONFIG_PATH)
    firebase_admin.initialize_app(cred, {
        "databaseURL": DATABASE_URL
    })


def fetch_sensor_data():
    ref = db.reference("sensor_data")
    data = ref.get()

    if not data:
        print("⚠️ Tidak ada data ditemukan di Firebase.")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "timestamp"
    df.reset_index(inplace=True)

    # rapikan urutan kolom
    expected_cols = [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "windSpeed",
        "irradiance",
        "voltage",
        "current",
        "power",
        "rssi"
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df = df[expected_cols]

    return df


def save_to_csv(df_new, filename=CSV_FILE):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if os.path.exists(filename):
        df_old = pd.read_csv(filename)

        # gabungkan
        df_combined = pd.concat([df_old, df_new])

        # hapus duplikat berdasarkan timestamp
        df_combined = df_combined.drop_duplicates(subset=["timestamp"])

        # urutkan waktu
        df_combined = df_combined.sort_values("timestamp")

        df_combined.to_csv(filename, index=False)

        print("✅ Data lama + baru berhasil digabung.")
    else:
        df_new.to_csv(filename, index=False)

        print("✅ File CSV baru dibuat.")


if __name__ == "__main__":
    df = fetch_sensor_data()

    print("Kolom dari Firebase:")
    print(df.columns)

    print(df.head())

    save_to_csv(df)