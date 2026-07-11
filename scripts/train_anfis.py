import os
import json
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_squared_error

from xanfis import AnfisRegressor


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SPLIT_PATH = os.path.join(BASE_DIR, "data", "processed", "anfis_split.npz")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "anfis_best_params.json")
GRID_RESULT_PATH = os.path.join(BASE_DIR, "data", "results", "anfis_grid_search.csv")

NAMA_VARIABEL = [
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "Irradiance",
]

# Hyperparameter arsitektur yang TETAP (dipakai lagi saat rekonstruksi
# model untuk prediksi di evaluate_anfis.py / skrip prediksi produksi).
MF_CLASS = "Gaussian"
TASK = "regression"  # single-output, karena tiap variabel punya model sendiri


def state_dict_path(var_name):
    return os.path.join(MODEL_DIR, "anfis_model_" + var_name + ".pt")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_predict(model, X):
    """Prediksi single-output + jaga-jaga NaN / keluar rentang scaler."""
    pred = np.asarray(model.predict(X)).reshape(-1)
    if np.isnan(pred).any():
        return None
    return np.clip(pred, 0, 1)


def main():
    print("\n===================================")
    print("MEMUAT DATA HASIL PREPROCESSING")
    print("===================================")

    data = np.load(SPLIT_PATH)
    X_train_scaled = data["X_train_scaled"]
    X_valid_scaled = data["X_valid_scaled"]
    y_train_scaled = data["y_train_scaled"]
    y_valid_scaled = data["y_valid_scaled"]

    print("Train :", len(X_train_scaled))
    print("Valid :", len(X_valid_scaled))

    input_dim = X_train_scaled.shape[1]

    # ============================
    # GRID SEARCH PER VARIABEL
    # ============================
    #
    # Setiap variabel cuaca dilatih sebagai model ANFIS TERPISAH
    # (single-output), bukan satu model gabungan untuk 5 output sekaligus,
    # supaya reg_lambda & num_rules bisa dioptimalkan khusus per variabel
    # (lihat diskusi sebelumnya: Temperature/WindSpeed butuh regularisasi
    # lebih ringan dibanding Pressure/Irradiance).
    #
    # Model TERBAIK dari tiap variabel disimpan bobotnya (state_dict)
    # secara langsung di sini -> TIDAK perlu dilatih ulang lagi di
    # evaluate_anfis.py atau skrip prediksi produksi manapun.

    reg_lambda_grid = [0.0001, 0.001, 0.01, 0.05]
    num_rules_grid = [3, 5, 7]

    print("\n===================================")
    print("GRID SEARCH PER VARIABEL")
    print("===================================")
    print(
        "Total kombinasi yang diuji: " +
        str(len(NAMA_VARIABEL) * len(reg_lambda_grid) * len(num_rules_grid)) +
        " (5 variabel x 3 num_rules x 4 reg_lambda). Mohon ditunggu."
    )

    all_grid_results = []
    best_params_per_var = {}

    os.makedirs(MODEL_DIR, exist_ok=True)

    for var_idx, var_name in enumerate(NAMA_VARIABEL):
        print("\n--- Grid search untuk variabel: " + var_name + " ---")

        y_train_i = y_train_scaled[:, var_idx]
        y_valid_i = y_valid_scaled[:, var_idx]

        best_score = np.inf
        best_params = None
        best_model_obj = None  # simpan objek model terbaik untuk variabel ini

        for nr in num_rules_grid:
            for rl in reg_lambda_grid:
                model_gs = AnfisRegressor(
                    num_rules=nr,
                    mf_class=MF_CLASS,
                    reg_lambda=rl,
                    epochs=200,
                    batch_size=128,
                    optim="Adam",
                    optim_params={"lr": 0.005},
                    early_stopping=True,
                    n_patience=15,
                    epsilon=0.0001,
                    valid_rate=0.15,
                    verbose=False
                )

                try:
                    model_gs.fit(X_train_scaled, y_train_i)
                    pred_valid = safe_predict(model_gs, X_valid_scaled)

                    if pred_valid is None:
                        score = np.inf
                    else:
                        score = rmse(y_valid_i, pred_valid)

                except Exception as e:
                    score = np.inf
                    print(
                        "  [gagal] " + var_name +
                        " num_rules=" + str(nr) +
                        " reg_lambda=" + str(rl) +
                        " -> " + str(e)
                    )

                all_grid_results.append({
                    "variabel": var_name,
                    "num_rules": nr,
                    "reg_lambda": rl,
                    "valid_rmse_normalized": round(score, 4) if np.isfinite(score) else None,
                })

                print(
                    "  num_rules=" + str(nr) +
                    "  reg_lambda=" + str(rl) +
                    "  valid_RMSE(norm)=" + format(score, ".4f")
                )

                if score < best_score:
                    best_score = score
                    best_params = {"num_rules": nr, "reg_lambda": rl}
                    best_model_obj = model_gs

        print(
            "-> Terbaik " + var_name + ": " + str(best_params) +
            "  (valid_RMSE norm=" + format(best_score, ".4f") + ")"
        )
        best_params_per_var[var_name] = best_params

        # ============================
        # SIMPAN BOBOT MODEL TERBAIK (state_dict, BUKAN pickle objek utuh)
        # ============================
        torch.save(
            best_model_obj.network.state_dict(),
            state_dict_path(var_name)
        )
        print("   Bobot model disimpan di:", state_dict_path(var_name))

    # ============================
    # SIMPAN HASIL
    # ============================
    os.makedirs(os.path.dirname(GRID_RESULT_PATH), exist_ok=True)
    pd.DataFrame(all_grid_results).to_csv(GRID_RESULT_PATH, index=False)

    # best_params.json menyimpan num_rules & reg_lambda per variabel,
    # DITAMBAH info arsitektur tetap (input_dim, mf_class, task) supaya
    # evaluate_anfis.py / skrip prediksi bisa merekonstruksi CustomANFIS
    # persis sama seperti saat training, sebelum memuat state_dict-nya.
    output_json = {
        "input_dim": int(input_dim),
        "mf_class": MF_CLASS,
        "task": TASK,
        "per_variabel": best_params_per_var,
    }
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(output_json, f, indent=2)

    print("\n===================================")
    print("RINGKASAN PARAMETER TERBAIK PER VARIABEL")
    print("===================================")
    for var_name, params in best_params_per_var.items():
        print("  " + var_name.ljust(12) + ": " + str(params))

    print("\nModel (state_dict) tersimpan di folder:", MODEL_DIR)
    print("Parameter arsitektur disimpan di:", BEST_PARAMS_PATH)
    print("Tabel grid search lengkap disimpan di:", GRID_RESULT_PATH)


if __name__ == "__main__":
    main()