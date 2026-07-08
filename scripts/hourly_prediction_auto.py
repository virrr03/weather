import subprocess
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

DETAIL_LOG_FILE = LOG_DIR / "hourly_prediction.log"
SUCCESS_LOG_FILE = LOG_DIR / "job_success.log"


STEPS = [
    ("Fetch data hourly", "fetch_data.py"),
    ("Kalibrasi data hourly", "calibration.py"),
    ("Preprocessing data hourly", "preprocess.py"),
    ("Prediksi 24 jam hourly", "predict_24h.py"),
]


def now_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_detail_log(message):
    text = f"[{now_time()}] {message}"
    print(text)

    with open(DETAIL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def write_success_log(step_name):
    text = f"{now_time()} | BERHASIL | {step_name}"

    with open(SUCCESS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def run_script(step_name, script_name):
    script_path = BASE_DIR / "scripts" / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {script_path}")

    write_detail_log(f"Mulai menjalankan {step_name}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        text=True,
        capture_output=True
    )

    if result.stdout:
        write_detail_log(result.stdout)

    if result.stderr:
        write_detail_log("ERROR/WARNING:")
        write_detail_log(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{step_name} gagal dijalankan")

    write_detail_log(f"Selesai menjalankan {step_name}")
    write_success_log(step_name)


def main():
    write_detail_log("====================================")
    write_detail_log("HOURLY PREDICTION DIMULAI")
    write_detail_log("====================================")

    try:
        for step_name, script_name in STEPS:
            run_script(step_name, script_name)

        write_detail_log("✅ Prediksi 24 jam hourly berhasil diperbarui")
        write_success_log("Hourly prediction selesai")

    except Exception as e:
        write_detail_log(f"❌ Hourly prediction gagal: {e}")


if __name__ == "__main__":
    main()