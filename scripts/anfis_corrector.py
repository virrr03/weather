import os
import json
import joblib
import numpy as np
import torch

from xanfis.models.base_anfis import CustomANFIS


class AnfisHybridCorrector:
    """
    Mengoreksi hasil prediksi RNN (dalam satuan ASLI/fisik: C, %, hPa, m/s, lux)
    memakai 5 model ANFIS final (satu per variabel), sesuai pipeline
    preprocess_anfis.py -> train_anfis.py -> evaluate_anfis.py:

        RNN pred (satuan asli)
            --[anfis_x_scaler.transform]--> 0-1
            --[CustomANFIS per variabel, direkonstruksi dari state_dict]--> 0-1
            --[anfis_y_scaler.inverse_transform]--> hasil koreksi (satuan asli)

    PENTING: model ANFIS TIDAK dimuat lewat AnfisRegressor.load_model()/pickle,
    karena itu bermasalah untuk CustomANFIS (torch.nn.Module) yang menyimpan
    referensi method private sebagai atribut instance -- gagal direkonstruksi
    di proses baru. Sebagai gantinya, hanya bobot (state_dict) yang disimpan
    saat training, dan di sini arsitektur CustomANFIS dibangun ulang secara
    manual dari anfis_best_params.json sebelum bobotnya dimuat.

    File yang dibutuhkan di <anfis_model_dir> (dihasilkan oleh train_anfis.py):
        - anfis_best_params.json   (arsitektur + num_rules/reg_lambda per variabel)
        - anfis_model_<Variabel>.pt  (state_dict, satu per variabel)

    File yang dibutuhkan di <anfis_scaler_dir> (dihasilkan oleh preprocess_anfis.py):
        - anfis_x_scaler.pkl
        - anfis_y_scaler.pkl

    Kalau scaler, best_params.json, atau salah satu file .pt tidak ditemukan,
    corrector otomatis fallback ke nilai RNN asli (tidak crash), supaya
    predict_7days.py / predict_24h.py tetap bisa jalan meski ANFIS belum
    lengkap dilatih.
    """

    NAMA_VARIABEL = ["Temperature", "Humidity", "Pressure", "WindSpeed", "Irradiance"]

    def __init__(self, anfis_model_dir, anfis_scaler_dir):
        self.anfis_model_dir = anfis_model_dir

        x_scaler_path = os.path.join(anfis_scaler_dir, "anfis_x_scaler.pkl")
        y_scaler_path = os.path.join(anfis_scaler_dir, "anfis_y_scaler.pkl")

        self.x_scaler = joblib.load(x_scaler_path) if os.path.exists(x_scaler_path) else None
        self.y_scaler = joblib.load(y_scaler_path) if os.path.exists(y_scaler_path) else None

        if self.x_scaler is None or self.y_scaler is None:
            print(
                "WARNING: anfis_x_scaler.pkl / anfis_y_scaler.pkl tidak ditemukan di "
                + anfis_scaler_dir + ". Koreksi ANFIS dinonaktifkan, "
                "prediksi memakai output RNN murni."
            )

        # ============================
        # MUAT ARSITEKTUR (best_params.json)
        # ============================
        best_params_path = os.path.join(anfis_model_dir, "anfis_best_params.json")
        self.arch_info = None
        self.best_params_per_var = {}

        if os.path.exists(best_params_path):
            try:
                with open(best_params_path, "r") as f:
                    info = json.load(f)
                self.arch_info = {
                    "input_dim": info["input_dim"],
                    "mf_class": info["mf_class"],
                    "task": info["task"],
                }
                self.best_params_per_var = info["per_variabel"]
            except Exception as e:
                print("WARNING: Gagal membaca anfis_best_params.json: " + str(e))
        else:
            print(
                "WARNING: anfis_best_params.json tidak ditemukan di "
                + anfis_model_dir + ". Koreksi ANFIS dinonaktifkan."
            )

        # ============================
        # MUAT SETIAP MODEL (state_dict -> CustomANFIS)
        # ============================
        self.models = {}

        for var_name in self.NAMA_VARIABEL:
            self.models[var_name] = None

            if self.arch_info is None or var_name not in self.best_params_per_var:
                print(
                    "WARNING: Parameter untuk " + var_name + " tidak tersedia. "
                    "Variabel ini dilewati (dipakai nilai RNN asli)."
                )
                continue

            state_dict_path = os.path.join(
                anfis_model_dir, "anfis_model_" + var_name + ".pt"
            )

            if not os.path.exists(state_dict_path):
                print(
                    "WARNING: Bobot model ANFIS untuk " + var_name +
                    " tidak ditemukan di " + state_dict_path +
                    ". Variabel ini dilewati (dipakai nilai RNN asli)."
                )
                continue

            try:
                params = self.best_params_per_var[var_name]
                network = CustomANFIS(
                    input_dim=self.arch_info["input_dim"],
                    num_rules=params["num_rules"],
                    output_dim=1,
                    mf_class=self.arch_info["mf_class"],
                    task=self.arch_info["task"],
                    reg_lambda=params["reg_lambda"],
                )
                state_dict = torch.load(state_dict_path, map_location="cpu")
                network.load_state_dict(state_dict)
                network.eval()

                self.models[var_name] = network
                print("OK: Model ANFIS " + var_name + " berhasil dimuat dari " + state_dict_path)
            except Exception as e:
                print("WARNING: Gagal memuat model ANFIS " + var_name + ": " + str(e))
                self.models[var_name] = None

    def is_ready(self):
        scalers_ok = self.x_scaler is not None and self.y_scaler is not None
        any_model = any(m is not None for m in self.models.values())
        return scalers_ok and any_model

    def correct(self, rnn_pred_original):
        """
        Parameters
        ----------
        rnn_pred_original : array-like, shape (1, 5)
            Prediksi RNN dalam SATUAN ASLI (hasil scaler.inverse_transform RNN),
            urutan kolom: [temperature, humidity, pressure, windSpeed, irradiance].

        Returns
        -------
        np.ndarray, shape (1, 5)
            Hasil koreksi ANFIS dalam satuan asli. Variabel yang modelnya
            tidak tersedia akan tetap memakai nilai RNN asli (fallback).
        """
        rnn_pred_original = np.asarray(rnn_pred_original, dtype=float).reshape(1, -1)

        if not self.is_ready():
            return rnn_pred_original.copy()

        X_scaled = self.x_scaler.transform(rnn_pred_original)
        X_scaled = np.clip(X_scaled, 0, 1)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        preds_scaled = np.full((1, len(self.NAMA_VARIABEL)), 0.5)
        missing_idx = []

        for idx, var_name in enumerate(self.NAMA_VARIABEL):
            network = self.models.get(var_name)

            if network is None:
                missing_idx.append(idx)
                continue

            try:
                with torch.no_grad():
                    pred_i = network(X_tensor).numpy().reshape(-1)
                preds_scaled[0, idx] = float(np.clip(pred_i[0], 0, 1))
            except Exception as e:
                print(
                    "WARNING: Gagal koreksi ANFIS untuk " + var_name +
                    ", dipakai nilai RNN asli: " + str(e)
                )
                missing_idx.append(idx)

        corrected = self.y_scaler.inverse_transform(preds_scaled)

        # Variabel yang gagal/tidak ada modelnya: pakai nilai RNN asli, bukan
        # hasil inverse_transform dari placeholder 0.5 (yang tidak berarti apa-apa).
        for idx in missing_idx:
            corrected[0, idx] = rnn_pred_original[0, idx]

        return corrected