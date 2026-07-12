import os
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import torch
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # aman untuk dijalankan tanpa display (headless)
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from xanfis.models.base_anfis import CustomANFIS

from evaluate import evaluate_all


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SPLIT_PATH = os.path.join(BASE_DIR, "data", "processed", "anfis_split.npz")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "anfis_best_params.json")

# Folder induk semua hasil evaluasi. Setiap kali skrip dijalankan, dibuat
# subfolder baru bertimestamp (mis. run_20260711_153045) supaya hasil dari
# run-run sebelumnya tidak tertimpa. Folder "latest" selalu berisi salinan
# hasil dari run paling baru, untuk akses cepat tanpa perlu tahu nama
# foldernya.
RESULT_ROOT_DIR = os.path.join(BASE_DIR, "data", "results")
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join(RESULT_ROOT_DIR, "run_" + RUN_TIMESTAMP)
LATEST_DIR = os.path.join(RESULT_ROOT_DIR, "latest")

COMPARISON_PATH = os.path.join(RESULT_DIR, "rnn_vs_anfis_comparison.csv")
PREDICTIONS_PATH = os.path.join(RESULT_DIR, "anfis_predictions.csv")
EVALUATION_METRICS_PATH = os.path.join(RESULT_DIR, "evaluation_metrics.json")
EVALUATION_METRICS_RNN_PATH = os.path.join(RESULT_DIR, "evaluation_metrics_rnn_only.json")

# Folder khusus untuk menyimpan semua grafik (di dalam folder run ini)
PLOT_DIR = os.path.join(RESULT_DIR, "plots")

NAMA_VARIABEL = [
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "Irradiance",
]

# Nama kolom versi lowercase, dipakai khusus untuk key di evaluation_metrics.json
# supaya cocok dengan FEATURE_COLS di predict_24h.py/predict_7days.py
# (yang membaca "per_feature" -> "temperature", "humidity", dst).
NAMA_VARIABEL_LOWER = {
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Pressure": "pressure",
    "WindSpeed": "windSpeed",
    "Irradiance": "irradiance",
}

# Satuan tiap variabel, dipakai untuk label sumbu Y pada grafik
SATUAN_VARIABEL = {
    "Temperature": "°C",
    "Humidity": "%",
    "Pressure": "hPa",
    "WindSpeed": "m/s",
    "Irradiance": "W/m²",
}


def state_dict_path(var_name):
    return os.path.join(MODEL_DIR, "anfis_model_" + var_name + ".pt")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_mape(y_true, y_pred, eps=1e-6):
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return np.nan
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def accuracy_from_mape(mape):
    """
    Accuracy = 100 - MAPE, dibatasi ke rentang [0, 100].

    Rumus ini SAMA PERSIS dengan accuracy_from_mape() di scripts/evaluate.py
    (evaluasi RNN), supaya Accuracy hasil ANFIS dan Accuracy hasil RNN
    bisa dibandingkan langsung apple-to-apple. Keduanya juga sama-sama
    dihitung dari data yang sudah dalam skala asli (bukan skala 0-1).
    """
    accuracy = 100 - mape
    accuracy = max(0, min(accuracy, 100))
    return round(float(accuracy), 2)


def hitung_metrik(y_true, y_pred):
    mape = round(safe_mape(y_true, y_pred), 2)
    return {
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "MAPE (%)": mape,
        "Accuracy": accuracy_from_mape(mape),
        "R2": round(r2_score(y_true, y_pred), 2),
    }


def inverse_transform_column(scaler, col_index, values_scaled, n_cols):
    """
    y_scaler dilatih untuk 5 kolom sekaligus (MinMaxScaler per-kolom
    independen). Untuk balikkan skala satu kolom saja, isi kolom lain
    dengan nilai apapun (dibuang), lalu ambil kolom yang relevan.
    """
    dummy = np.zeros((len(values_scaled), n_cols))
    dummy[:, col_index] = values_scaled
    inv = scaler.inverse_transform(dummy)
    return inv[:, col_index]


def load_anfis_model(var_name, params, arch_info):
    """
    Rekonstruksi CustomANFIS langsung dari state_dict (BUKAN dari pickle
    objek AnfisRegressor utuh, karena itu bermasalah - lihat catatan di
    train_anfis.py). Arsitektur direkonstruksi persis sama seperti saat
    training: input_dim, mf_class, task tetap; num_rules & reg_lambda
    sesuai hasil grid search untuk variabel ini.
    """
    network = CustomANFIS(
        input_dim=arch_info["input_dim"],
        num_rules=params["num_rules"],
        output_dim=1,
        mf_class=arch_info["mf_class"],
        task=arch_info["task"],
        reg_lambda=params["reg_lambda"],
    )
    state_dict = torch.load(state_dict_path(var_name), map_location="cpu")
    network.load_state_dict(state_dict)
    network.eval()
    return network


def predict_with_network(network, X_scaled):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred = network(X_tensor).numpy().reshape(-1)
    return pred


# ==========================================================
# FUNGSI-FUNGSI PLOT
# ==========================================================

def plot_time_series(var_name, y_true, y_pred_rnn, y_pred_anfis, n_titik=200):
    n = min(n_titik, len(y_true))
    satuan = SATUAN_VARIABEL.get(var_name, "")

    plt.figure(figsize=(11, 4.5))
    plt.plot(y_true[:n], label="Aktual", color="black", linewidth=1.8)
    plt.plot(y_pred_rnn[:n], label="RNN saja", color="tab:orange", linestyle="--", linewidth=1.2)
    plt.plot(y_pred_anfis[:n], label="RNN + ANFIS", color="tab:blue", linewidth=1.4)
    plt.title(f"Perbandingan Prediksi vs Aktual - {var_name}")
    plt.xlabel("Indeks data uji")
    plt.ylabel(f"{var_name} ({satuan})" if satuan else var_name)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"timeseries_{var_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_scatter(var_name, y_true, y_pred_rnn, y_pred_anfis):
    satuan = SATUAN_VARIABEL.get(var_name, "")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)

    batas_min = min(y_true.min(), y_pred_rnn.min(), y_pred_anfis.min())
    batas_max = max(y_true.max(), y_pred_rnn.max(), y_pred_anfis.max())

    for ax, pred, judul, warna in [
        (axes[0], y_pred_rnn, "RNN saja", "tab:orange"),
        (axes[1], y_pred_anfis, "RNN + ANFIS", "tab:blue"),
    ]:
        ax.scatter(y_true, pred, alpha=0.5, s=14, color=warna, edgecolor="none")
        ax.plot([batas_min, batas_max], [batas_min, batas_max], color="gray", linestyle="--", linewidth=1)
        ax.set_title(judul)
        ax.set_xlabel(f"Aktual ({satuan})" if satuan else "Aktual")
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel(f"Prediksi ({satuan})" if satuan else "Prediksi")
    fig.suptitle(f"Scatter Prediksi vs Aktual - {var_name}")
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"scatter_{var_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_ringkasan_metrik(hasil_perbandingan_df):
    df = hasil_perbandingan_df[
        hasil_perbandingan_df["Variabel"] != "RATA-RATA (macro-average)"
    ].copy()

    x = np.arange(len(df))
    lebar = 0.35

    metrik_list = [
        ("RMSE", "RNN_RMSE", "ANFIS_RMSE"),
        ("MAE", "RNN_MAE", "ANFIS_MAE"),
        ("MAPE (%)", "RNN_MAPE", "ANFIS_MAPE"),
        ("R2", "RNN_R2", "ANFIS_R2"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (judul, kolom_rnn, kolom_anfis) in zip(axes, metrik_list):
        ax.bar(x - lebar / 2, df[kolom_rnn], width=lebar, label="RNN saja", color="tab:orange")
        ax.bar(x + lebar / 2, df[kolom_anfis], width=lebar, label="RNN + ANFIS", color="tab:blue")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Variabel"], rotation=30, ha="right")
        ax.set_title(judul)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    fig.suptitle("Ringkasan Perbandingan Metrik: RNN saja vs RNN + ANFIS", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "ringkasan_metrik.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_akurasi_persen(hasil_perbandingan_df):
    df = hasil_perbandingan_df[
        hasil_perbandingan_df["Variabel"] != "RATA-RATA (macro-average)"
    ].copy()

    akurasi_rnn = df["RNN_Accuracy"]
    akurasi_anfis = df["ANFIS_Accuracy"]

    x = np.arange(len(df))
    lebar = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - lebar / 2, akurasi_rnn, width=lebar, label="RNN saja", color="tab:orange")
    plt.bar(x + lebar / 2, akurasi_anfis, width=lebar, label="RNN + ANFIS", color="tab:blue")
    plt.xticks(x, df["Variabel"], rotation=20, ha="right")
    plt.ylabel("Akurasi (%) = 100 - MAPE")
    plt.title("Akurasi Prediksi per Variabel")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    for i, (a_rnn, a_anfis) in enumerate(zip(akurasi_rnn, akurasi_anfis)):
        plt.text(i - lebar / 2, a_rnn + 1, f"{a_rnn:.1f}%", ha="center", fontsize=8)
        plt.text(i + lebar / 2, a_anfis + 1, f"{a_anfis:.1f}%", ha="center", fontsize=8)

    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "akurasi_persen.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    print("\n===================================")
    print("MEMUAT DATA & MODEL (state_dict, tanpa latih ulang)")
    print("===================================")

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("Hasil run ini akan disimpan di folder bertimestamp:")
    print(" ", RESULT_DIR)

    data = np.load(SPLIT_PATH)
    X_test_scaled = data["X_test_scaled"]
    X_test_orig = data["X_test_orig"]      # ini = prediksi RNN (skala asli)
    y_test_orig = data["y_test_orig"]      # ini = data aktual (skala asli)

    y_scaler = joblib.load(os.path.join(MODEL_DIR, "anfis_y_scaler.pkl"))

    with open(BEST_PARAMS_PATH, "r") as f:
        best_params_json = json.load(f)

    arch_info = {
        "input_dim": best_params_json["input_dim"],
        "mf_class": best_params_json["mf_class"],
        "task": best_params_json["task"],
    }
    best_params_per_var = best_params_json["per_variabel"]

    n_cols = len(NAMA_VARIABEL)
    y_pred_rnn = X_test_orig
    y_pred_anfis = np.zeros_like(y_test_orig)

    hasil_perbandingan = []
    r2_rnn_list, r2_anfis_list = [], []
    mape_rnn_list, mape_anfis_list = [], []
    plot_paths = []

    print("\n===================================")
    print("PERBANDINGAN: RNN SAJA vs RNN + ANFIS")
    print("===================================")

    for i, var_name in enumerate(NAMA_VARIABEL):
        params = best_params_per_var[var_name]
        network = load_anfis_model(var_name, params, arch_info)

        pred_scaled = predict_with_network(network, X_test_scaled)
        if np.isnan(pred_scaled).any():
            raise ValueError("Prediksi " + var_name + " mengandung NaN")
        pred_scaled = np.clip(pred_scaled, 0, 1)

        pred_orig = inverse_transform_column(y_scaler, i, pred_scaled, n_cols)
        y_pred_anfis[:, i] = pred_orig

        y_true_i = y_test_orig[:, i]
        metrik_rnn = hitung_metrik(y_true_i, y_pred_rnn[:, i])
        metrik_anfis = hitung_metrik(y_true_i, pred_orig)
        std_actual = round(float(np.std(y_true_i)), 2)

        print("\n" + var_name + " (std aktual = " + str(std_actual) + ", params=" + str(params) + ")")
        print(
            "  RNN saja    : RMSE=" + str(metrik_rnn["RMSE"]) +
            "  MAE=" + str(metrik_rnn["MAE"]) +
            "  MAPE=" + str(metrik_rnn["MAPE (%)"]) + "%" +
            "  Accuracy=" + str(metrik_rnn["Accuracy"]) + "%" +
            "  R2=" + str(metrik_rnn["R2"])
        )
        print(
            "  RNN + ANFIS : RMSE=" + str(metrik_anfis["RMSE"]) +
            "  MAE=" + str(metrik_anfis["MAE"]) +
            "  MAPE=" + str(metrik_anfis["MAPE (%)"]) + "%" +
            "  Accuracy=" + str(metrik_anfis["Accuracy"]) + "%" +
            "  R2=" + str(metrik_anfis["R2"])
        )

        if std_actual < metrik_anfis["RMSE"]:
            print(
                "  -> Catatan: std aktual variabel ini kecil, R2 kurang "
                "representatif. Acuan utama: RMSE/MAE/MAPE."
            )

        r2_rnn_list.append(r2_score(y_true_i, y_pred_rnn[:, i]))
        r2_anfis_list.append(r2_score(y_true_i, pred_orig))
        mape_rnn_list.append(safe_mape(y_true_i, y_pred_rnn[:, i]))
        mape_anfis_list.append(safe_mape(y_true_i, pred_orig))

        hasil_perbandingan.append({
            "Variabel": var_name,
            "num_rules": params["num_rules"],
            "reg_lambda": params["reg_lambda"],
            "Std_Aktual": std_actual,
            "RNN_RMSE": metrik_rnn["RMSE"],
            "RNN_MAE": metrik_rnn["MAE"],
            "RNN_MAPE": metrik_rnn["MAPE (%)"],
            "RNN_Accuracy": metrik_rnn["Accuracy"],
            "RNN_R2": metrik_rnn["R2"],
            "ANFIS_RMSE": metrik_anfis["RMSE"],
            "ANFIS_MAE": metrik_anfis["MAE"],
            "ANFIS_MAPE": metrik_anfis["MAPE (%)"],
            "ANFIS_Accuracy": metrik_anfis["Accuracy"],
            "ANFIS_R2": metrik_anfis["R2"],
        })

        # ---- grafik per variabel ----
        p1 = plot_time_series(var_name, y_true_i, y_pred_rnn[:, i], pred_orig)
        p2 = plot_scatter(var_name, y_true_i, y_pred_rnn[:, i], pred_orig)
        plot_paths.extend([p1, p2])

    # ============================
    # RINGKASAN (macro-average R2 & MAPE, tidak menggabungkan skala RMSE/MAE)
    # ============================
    print("\n===================================")
    print("RINGKASAN (macro-average, tidak menggabungkan satuan berbeda)")
    print("===================================")

    ringkasan_rnn = {
        "R2 (rata-rata)": round(float(np.mean(r2_rnn_list)), 2),
        "MAPE (%) (rata-rata)": round(float(np.mean(mape_rnn_list)), 2),
        "Accuracy (%) (rata-rata)": accuracy_from_mape(float(np.mean(mape_rnn_list))),
    }
    ringkasan_anfis = {
        "R2 (rata-rata)": round(float(np.mean(r2_anfis_list)), 2),
        "MAPE (%) (rata-rata)": round(float(np.mean(mape_anfis_list)), 2),
        "Accuracy (%) (rata-rata)": accuracy_from_mape(float(np.mean(mape_anfis_list))),
    }

    print("RNN saja    :", ringkasan_rnn)
    print("RNN + ANFIS :", ringkasan_anfis)

    hasil_perbandingan.append({
        "Variabel": "RATA-RATA (macro-average)",
        "num_rules": None,
        "reg_lambda": None,
        "Std_Aktual": None,
        "RNN_RMSE": None,
        "RNN_MAE": None,
        "RNN_MAPE": ringkasan_rnn["MAPE (%) (rata-rata)"],
        "RNN_Accuracy": ringkasan_rnn["Accuracy (%) (rata-rata)"],
        "RNN_R2": ringkasan_rnn["R2 (rata-rata)"],
        "ANFIS_RMSE": None,
        "ANFIS_MAE": None,
        "ANFIS_MAPE": ringkasan_anfis["MAPE (%) (rata-rata)"],
        "ANFIS_Accuracy": ringkasan_anfis["Accuracy (%) (rata-rata)"],
        "ANFIS_R2": ringkasan_anfis["R2 (rata-rata)"],
    })

    df_perbandingan = pd.DataFrame(hasil_perbandingan)

    # ============================
    # SIMPAN HASIL (CSV)
    # ============================
    df_perbandingan.to_csv(COMPARISON_PATH, index=False)
    print("\nTabel perbandingan RNN vs RNN+ANFIS disimpan di:")
    print(COMPARISON_PATH)

    prediksi_df = pd.DataFrame({
        "temperature_actual": y_test_orig[:, 0],
        "temperature_pred_rnn": y_pred_rnn[:, 0],
        "temperature_pred_anfis": y_pred_anfis[:, 0],
        "humidity_actual": y_test_orig[:, 1],
        "humidity_pred_rnn": y_pred_rnn[:, 1],
        "humidity_pred_anfis": y_pred_anfis[:, 1],
        "pressure_actual": y_test_orig[:, 2],
        "pressure_pred_rnn": y_pred_rnn[:, 2],
        "pressure_pred_anfis": y_pred_anfis[:, 2],
        "wind_actual": y_test_orig[:, 3],
        "wind_pred_rnn": y_pred_rnn[:, 3],
        "wind_pred_anfis": y_pred_anfis[:, 3],
        "irradiance_actual": y_test_orig[:, 4],
        "irradiance_pred_rnn": y_pred_rnn[:, 4],
        "irradiance_pred_anfis": y_pred_anfis[:, 4],
    })
    prediksi_df.to_csv(PREDICTIONS_PATH, index=False)
    print("Hasil prediksi lengkap disimpan di:")
    print(PREDICTIONS_PATH)

    # ============================
    # EVALUATION_METRICS.JSON (format sama seperti scripts/evaluate.py,
    # supaya predict_7days.py/predict_24h.py bisa baca akurasi model
    # HYBRID (RNN+ANFIS) yang sebenarnya, bukan akurasi RNN saja)
    # ============================
    # Dipakai fungsi evaluate_all() yang SAMA dari scripts/evaluate.py
    # (bukan menulis ulang logikanya di sini), supaya kalau nanti formula
    # metriknya diubah, cukup diubah di satu tempat (evaluate.py) dan
    # otomatis konsisten di kedua laporan (RNN & ANFIS).
    feature_names_lower = [NAMA_VARIABEL_LOWER[v] for v in NAMA_VARIABEL]

    metrics_anfis = evaluate_all(
        y_test_orig, y_pred_anfis, feature_names_lower, accuracy_method="mape"
    )
    metrics_rnn_only = evaluate_all(
        y_test_orig, y_pred_rnn, feature_names_lower, accuracy_method="mape"
    )

    with open(EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_anfis, f, indent=2)
    with open(EVALUATION_METRICS_RNN_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_rnn_only, f, indent=2)

    print("\n===================================")
    print("EVALUATION METRICS (untuk dashboard & laporan)")
    print("===================================")
    print("Model HYBRID (RNN+ANFIS) - overall:", metrics_anfis["overall"])
    print("Disimpan di:", EVALUATION_METRICS_PATH)
    print("(Perbandingan RNN saja disimpan terpisah di:", EVALUATION_METRICS_RNN_PATH, ")")

    # ============================
    # GRAFIK RINGKASAN (metrik + akurasi %)
    # ============================
    p_ringkasan = plot_ringkasan_metrik(df_perbandingan)
    p_akurasi = plot_akurasi_persen(df_perbandingan)
    plot_paths.extend([p_ringkasan, p_akurasi])

    print("\n===================================")
    print("GRAFIK DISIMPAN DI:")
    print("===================================")
    for p in plot_paths:
        print(" -", p)

    # ============================
    # SALIN HASIL RUN INI KE FOLDER "latest" (akses cepat, selalu terbaru)
    # ============================
    # evaluation_metrics.json di folder "latest" inilah yang dibaca
    # predict_7days.py / predict_24h.py untuk field model_accuracy_percent.
    if os.path.exists(LATEST_DIR):
        shutil.rmtree(LATEST_DIR)
    shutil.copytree(RESULT_DIR, LATEST_DIR)

    print("\n===================================")
    print("RIWAYAT & AKSES CEPAT")
    print("===================================")
    print("Hasil run ini tersimpan permanen (tidak akan tertimpa) di:")
    print(" ", RESULT_DIR)
    print("Salinan hasil terbaru juga tersedia di:")
    print(" ", LATEST_DIR)
    print(
        "\nPENTING: predict_7days.py/predict_24h.py sekarang akan membaca "
        "akurasi model HYBRID (RNN+ANFIS) dari file ini, bukan lagi akurasi "
        "RNN saja."
    )


if __name__ == "__main__":
    main()