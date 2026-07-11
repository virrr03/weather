import os
import json
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from xanfis.models.base_anfis import CustomANFIS


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SPLIT_PATH = os.path.join(BASE_DIR, "data", "processed", "anfis_split.npz")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "anfis_best_params.json")

RESULT_DIR = os.path.join(BASE_DIR, "data", "results")
COMPARISON_PATH = os.path.join(RESULT_DIR, "rnn_vs_anfis_comparison.csv")
PREDICTIONS_PATH = os.path.join(RESULT_DIR, "anfis_predictions.csv")

NAMA_VARIABEL = [
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "Irradiance",
]


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


def hitung_metrik(y_true, y_pred):
    return {
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "MAPE (%)": round(safe_mape(y_true, y_pred), 2),
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


def main():
    print("\n===================================")
    print("MEMUAT DATA & MODEL (state_dict, tanpa latih ulang)")
    print("===================================")

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
            "  R2=" + str(metrik_rnn["R2"])
        )
        print(
            "  RNN + ANFIS : RMSE=" + str(metrik_anfis["RMSE"]) +
            "  MAE=" + str(metrik_anfis["MAE"]) +
            "  MAPE=" + str(metrik_anfis["MAPE (%)"]) + "%" +
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
            "RNN_R2": metrik_rnn["R2"],
            "ANFIS_RMSE": metrik_anfis["RMSE"],
            "ANFIS_MAE": metrik_anfis["MAE"],
            "ANFIS_MAPE": metrik_anfis["MAPE (%)"],
            "ANFIS_R2": metrik_anfis["R2"],
        })

    # ============================
    # RINGKASAN (macro-average R2 & MAPE, tidak menggabungkan skala RMSE/MAE)
    # ============================
    print("\n===================================")
    print("RINGKASAN (macro-average, tidak menggabungkan satuan berbeda)")
    print("===================================")

    ringkasan_rnn = {
        "R2 (rata-rata)": round(float(np.mean(r2_rnn_list)), 2),
        "MAPE (%) (rata-rata)": round(float(np.mean(mape_rnn_list)), 2),
    }
    ringkasan_anfis = {
        "R2 (rata-rata)": round(float(np.mean(r2_anfis_list)), 2),
        "MAPE (%) (rata-rata)": round(float(np.mean(mape_anfis_list)), 2),
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
        "RNN_R2": ringkasan_rnn["R2 (rata-rata)"],
        "ANFIS_RMSE": None,
        "ANFIS_MAE": None,
        "ANFIS_MAPE": ringkasan_anfis["MAPE (%) (rata-rata)"],
        "ANFIS_R2": ringkasan_anfis["R2 (rata-rata)"],
    })

    # ============================
    # SIMPAN HASIL
    # ============================
    os.makedirs(RESULT_DIR, exist_ok=True)

    pd.DataFrame(hasil_perbandingan).to_csv(COMPARISON_PATH, index=False)
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


if __name__ == "__main__":
    main()