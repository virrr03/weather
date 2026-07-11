# scripts/calibration.py

import pandas as pd


# ============================
# KONFIGURASI KALIBRASI
# ============================
# scale & offset masih default (1.0 / 0.0) karena belum dilakukan
# kalibrasi terhadap alat ukur referensi/standar. Jika di kemudian hari
# dilakukan kalibrasi (membandingkan pembacaan sensor dengan alat ukur
# bersertifikat), nilai scale/offset di bawah ini yang perlu diisi.
#
# Fungsi-fungsi di file ini saat ini berperan sebagai:
# 1. Konversi satuan (contoh: Pa -> hPa untuk pressure)
# 2. Pembatasan (clamp) nilai ke rentang fisik yang wajar, sebagai
#    lapisan pengaman kedua SETELAH filter outlier di preprocess.py
CALIBRATION = {
    "temperature": {
        "scale": 1.0,
        "offset": 0.0
    },

    "humidity": {
        "scale": 1.0,
        "offset": 0.0
    },

    "pressure": {
        "scale": 1.0,
        "offset": 0.0
    },

    "windSpeed": {
        "scale": 1.0,
        "offset": 0.0
    },

    "irradiance": {
        "scale": 1.0,
        "offset": 0.0
    }
}

LIGHT_MAX_RAW = 1000.0


def safe_float(value, default=0.0):
    """
    Mengubah nilai menjadi float dengan aman.
    Kalau kosong, NaN, atau error, akan diganti default.
    """
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def apply_linear_calibration(value, sensor_name):
    """
    Kalibrasi linear:
    nilai_kalibrasi = nilai_sensor * scale + offset
    """
    value = safe_float(value)

    scale = CALIBRATION[sensor_name]["scale"]
    offset = CALIBRATION[sensor_name]["offset"]

    return (value * scale) + offset


def calibrate_temperature(value):
    """
    Kalibrasi suhu.
    Satuan input dan output: derajat Celsius.

    CATATAN: nilai di luar rentang wajar seharusnya SUDAH dibuang oleh
    filter outlier di preprocess.py sebelum sampai ke fungsi ini. Clamp
    di bawah ini hanya sebagai lapisan pengaman kedua (safety net),
    bukan mekanisme utama pembersihan data.
    """
    value = apply_linear_calibration(value, "temperature")

    # Batas aman untuk data suhu lingkungan lokal
    value = max(-10, min(value, 60))

    return round(value, 2)


def calibrate_humidity(value):
    """
    Kalibrasi kelembapan.
    Satuan input dan output: persen RH.
    """
    value = apply_linear_calibration(value, "humidity")

    # Kelembapan harus 0 sampai 100 persen
    value = max(0, min(value, 100))

    return round(value, 2)


def calibrate_pressure(value):
    """
    Kalibrasi tekanan udara.
    Jika sensor membaca Pa, otomatis diubah ke hPa.
    Contoh:
    101325 Pa -> 1013.25 hPa
    """
    value = safe_float(value)

    # Jika tekanan masih dalam Pa, ubah ke hPa
    if value > 2000:
        value = value / 100.0

    value = apply_linear_calibration(value, "pressure")

    # Batas wajar tekanan udara dalam hPa
    value = max(800, min(value, 1200))

    return round(value, 2)


def calibrate_wind_speed(value):
    """
    Kalibrasi kecepatan angin.
    Satuan input dan output: m/s.
    """
    value = apply_linear_calibration(value, "windSpeed")

    # Kecepatan angin tidak boleh negatif
    value = max(0, value)

    return round(value, 2)


def calibrate_light_intensity(value):
    """
    Kalibrasi intensitas cahaya (irradiance).
    Satuan input dan output: W/m^2 (hasil konversi dari pembacaan
    tegangan-arus panel surya via INA219, lihat rumus fotovoltaic
    G = V.I / (eta.A) pada BAB II).
    """
    value = safe_float(value)

    value = apply_linear_calibration(value, "irradiance")

    # batas sensor
    value = max(0, min(value, 1000))

    return round(value, 2)


def apply_sensor_calibration(df):
    """
    Fungsi utama untuk dipanggil dari preprocess.py.

    Input:
    DataFrame dari sensor_data.csv (idealnya SUDAH melewati filter
    outlier di preprocess.py terlebih dahulu, supaya clamp di sini
    tidak diam-diam mengganti nilai error sensor menjadi nilai lain
    yang terlihat wajar padahal bukan hasil pengukuran nyata).

    Output:
    DataFrame yang sudah dikalibrasi
    """
    df = df.copy()

    if "temperature" in df.columns:
        df["temperature"] = df["temperature"].apply(calibrate_temperature)

    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].apply(calibrate_humidity)

    if "pressure" in df.columns:
        df["pressure"] = df["pressure"].apply(calibrate_pressure)

    if "windSpeed" in df.columns:
        df["windSpeed"] = df["windSpeed"].apply(calibrate_wind_speed)

    if "irradiance" in df.columns:
        df["irradiance"] = df["irradiance"].apply(calibrate_light_intensity)

    return df


def print_active_calibration():
    """
    Menampilkan konfigurasi kalibrasi aktif.
    """
    print("\n=== KONFIGURASI KALIBRASI AKTIF ===")
    for sensor_name, config in CALIBRATION.items():
        print(
            f"{sensor_name}: "
            f"scale={config['scale']}, "
            f"offset={config['offset']}"
        )

    print(f"LIGHT_MAX_RAW: {LIGHT_MAX_RAW}")
    print("===================================\n")


if __name__ == "__main__":
    print_active_calibration()