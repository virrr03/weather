import subprocess
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

DETAIL_LOG_FILE = LOG_DIR / "daily_auto.log"
SUCCESS_LOG_FILE = LOG_DIR / "job_success.log"


STEPS = [
    ("Fetch data daily", "fetch_data.py"),
    ("Preprocessing data daily", "preprocess.py"),
    ("Training RNN", "train_rnn.py"),
    ("Evaluasi dan plot model", "run_evaluation.py"),
    ("generate rnn prediksi", "generate_rnn_predictions.py"),
    ("buat database anfis", "build_anfis_dataset.py"),
    ("preprcessing dataset untuk anfis", "preprocess_anfis.py"),
    ("training anfis", "train_anfis.py"),
    ("evaluasi model anfis", "evaluate_anfis.py"),
    ("Prediksi 7 hari", "predict_7days.py"),
    ("Prediksi 24 jam daily", "predict_24h.py"),
]


def now_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_detail_log(message):
    text = f"[{now_time()}] {message}"
    print(text)

    with open(DETAIL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def write_job_log(status, step_name):
    text = f"{now_time()} | {status} | {step_name}"

    with open(SUCCESS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def run_script(step_name, script_name):
    script_path = BASE_DIR / "scripts" / script_name

    if not script_path.exists():
        write_job_log("GAGAL", step_name)
        raise FileNotFoundError(f"File tidak ditemukan: {script_path}")

    write_detail_log(f"Mulai menjalankan {step_name}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        text=True,
        capture_output=True
    )

    if result.stdout:
        write_detail_log("OUTPUT:")
        write_detail_log(result.stdout)

    if result.stderr:
        write_detail_log("ERROR/WARNING:")
        write_detail_log(result.stderr)

    if result.returncode != 0:
        write_job_log("GAGAL", step_name)
        raise RuntimeError(f"{step_name} gagal dijalankan")

    write_detail_log(f"Selesai menjalankan {step_name}")
    write_job_log("BERHASIL", step_name)


def main():
    write_detail_log("====================================")
    write_detail_log("DAILY AUTO PIPELINE DIMULAI")
    write_detail_log("====================================")

    try:
        for step_name, script_name in STEPS:
            run_script(step_name, script_name)

        write_detail_log("✅ Semua proses harian berhasil dijalankan")
        write_job_log("BERHASIL", "Daily pipeline selesai")

    except Exception as e:
        write_detail_log(f"❌ Daily pipeline gagal: {e}")
        write_job_log("GAGAL", "Daily pipeline gagal")


if __name__ == "__main__":
    main()