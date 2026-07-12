# scripts/evaluate.py

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


EPSILON = 1e-8


def safe_mape(y_actual, y_pred):
    """
    MAPE aman untuk data yang mengandung nilai 0.
    Data aktual yang bernilai 0 atau sangat kecil tidak dihitung.
    """
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)

    mask = np.abs(y_actual) > EPSILON

    if not np.any(mask):
        return None

    mape = np.mean(
        np.abs((y_actual[mask] - y_pred[mask]) / y_actual[mask])
    ) * 100

    return round(float(mape), 4)


def accuracy_from_mae(mae):
    """
    PERINGATAN: formula ini HANYA valid kalau data sudah dinormalisasi 0-1.
    Accuracy = (1 - MAE) * 100 tidak bermakna untuk data satuan asli
    (Celsius, hPa, dst) -- MAE=2 (misal 2 derajat Celsius) akan dianggap
    "Accuracy -100%" walau sebenarnya prediksi itu cukup baik.

    Kalau memanggil fungsi ini, PASTIKAN mae dihitung dari data yang
    memang masih dalam rentang 0-1. Untuk evaluasi di satuan asli,
    pakai accuracy_from_mape() sebagai gantinya (lihat evaluate_original_scale).
    """
    accuracy = (1 - mae) * 100
    accuracy = max(0, min(accuracy, 100))
    return round(float(accuracy), 2)


def accuracy_from_mape(mape):
    """
    Accuracy = 100 - MAPE, dibatasi ke rentang [0, 100].

    Berbeda dengan accuracy_from_mae(), rumus ini SAH dipakai di satuan
    ASLI (bukan cuma 0-1) karena MAPE sendiri sudah berupa persentase
    error relatif -- tidak bergantung skala/satuan variabel.
    """
    if mape is None:
        return None
    accuracy = 100 - mape
    accuracy = max(0, min(accuracy, 100))
    return round(float(accuracy), 2)


def _cek_kemungkinan_masih_ternormalisasi(y_actual, nama_konteks=""):
    """
    Jaga-jaga: kalau SELURUH nilai y_actual ada di rentang [0, 1], kemungkinan
    besar data ini belum di-inverse_transform ke satuan asli. Ini bukan bukti
    pasti (ada variabel yang secara alami memang di 0-1), tapi cukup untuk
    peringatan dini supaya tidak keulang lagi kejadian evaluasi RNN yang
    ternyata masih di skala normalisasi.
    """
    y_actual = np.asarray(y_actual, dtype=float)
    if y_actual.size == 0:
        return
    if np.nanmin(y_actual) >= -1e-6 and np.nanmax(y_actual) <= 1.0 + 1e-6:
        print(
            "PERINGATAN" + (" (" + nama_konteks + ")" if nama_konteks else "") +
            ": seluruh nilai y_actual berada di rentang [0, 1]. "
            "Kalau ini BUKAN memang variabel yang secara alami di 0-1 "
            "(mis. probabilitas), kemungkinan besar data belum di-"
            "inverse_transform ke satuan asli sebelum dievaluasi -- "
            "metrik di bawah ini bisa menyesatkan (lihat diskusi Temperature/"
            "Pressure yang MAPE-nya meledak kalau dihitung di ruang "
            "ternormalisasi)."
        )


def evaluate_overall(y_actual, y_pred, accuracy_method="mae"):
    """
    Evaluasi gabungan seluruh parameter cuaca.

    PENTING: y_actual dan y_pred HARUS sudah dalam satuan ASLI (hasil
    scaler.inverse_transform), bukan data ternormalisasi 0-1. Kalau masih
    ternormalisasi, MAE/RMSE tidak bermakna secara fisis.

    accuracy_method : "mae" (default, HANYA valid untuk data 0-1) atau
        "mape" (valid untuk satuan APA PUN, termasuk satuan asli).
        Pakai "mape" kalau y_actual/y_pred dalam satuan asli.
    """
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)

    _cek_kemungkinan_masih_ternormalisasi(y_actual, "evaluate_overall")

    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mape = safe_mape(y_actual, y_pred)

    if accuracy_method == "mape":
        accuracy = accuracy_from_mape(mape)
    else:
        accuracy = accuracy_from_mae(mae)

    try:
        r2 = r2_score(y_actual, y_pred)
    except Exception:
        r2 = 0

    return {
        "MAE": round(float(mae), 4),
        "MAPE": mape,
        "RMSE": round(float(rmse), 4),
        "Accuracy": accuracy,
        "R2_Score": round(float(r2), 4)
    }


def evaluate_per_feature(y_actual, y_pred, feature_names, accuracy_method="mae"):
    """
    Evaluasi tiap parameter cuaca.

    PENTING: sama seperti evaluate_overall(), y_actual/y_pred harus
    sudah dalam satuan ASLI, bukan ternormalisasi. Lihat accuracy_method
    di evaluate_overall() untuk penjelasan pemilihan formula akurasi.
    """
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)

    results = {}

    for i, feature in enumerate(feature_names):
        actual_feature = y_actual[:, i]
        pred_feature = y_pred[:, i]

        mae = mean_absolute_error(actual_feature, pred_feature)
        rmse = np.sqrt(mean_squared_error(actual_feature, pred_feature))
        mape = safe_mape(actual_feature, pred_feature)

        if accuracy_method == "mape":
            accuracy = accuracy_from_mape(mape)
        else:
            accuracy = accuracy_from_mae(mae)

        try:
            r2 = r2_score(actual_feature, pred_feature)
        except Exception:
            r2 = 0

        results[feature] = {
            "MAE": round(float(mae), 4),
            "MAPE": mape,
            "RMSE": round(float(rmse), 4),
            "Accuracy": accuracy,
            "R2_Score": round(float(r2), 4)
        }

    return results


def evaluate_all(y_actual, y_pred, feature_names, accuracy_method="mae"):
    """
    Menghasilkan evaluasi gabungan dan evaluasi per parameter.

    PENTING: y_actual dan y_pred harus SUDAH dalam satuan asli.
    Kalau data Anda masih hasil scaler.transform() (rentang 0-1),
    pakai evaluate_all_original_scale() di bawah -- itu akan
    melakukan inverse_transform dulu secara otomatis sebelum
    menghitung metrik apapun, DAN otomatis pakai accuracy_method="mape".
    """
    return {
        "overall": evaluate_overall(y_actual, y_pred, accuracy_method=accuracy_method),
        "per_feature": evaluate_per_feature(y_actual, y_pred, feature_names, accuracy_method=accuracy_method)
    }


def evaluate_all_original_scale(y_actual_scaled, y_pred_scaled, feature_names, scaler):
    """
    Versi AMAN dari evaluate_all(): menerima data yang MASIH dalam skala
    hasil scaler.transform() (0-1), lalu melakukan scaler.inverse_transform()
    sendiri sebelum menghitung metrik apapun -- supaya tidak ada lagi
    kejadian evaluasi dihitung di ruang ternormalisasi seperti sebelumnya.
    Otomatis pakai accuracy_method="mape" karena hasilnya sudah satuan asli.

    Parameters
    ----------
    y_actual_scaled, y_pred_scaled : array shape (n_samples, n_features)
        Data HASIL scaler.transform() (rentang 0-1), urutan kolom harus
        sama dengan feature_names dan sama dengan urutan kolom saat
        scaler pertama kali di-fit.
    feature_names : list[str]
        Nama tiap kolom, urutannya harus sama dengan kolom di scaler.
    scaler : sklearn-like scaler
        Scaler yang SAMA persis dipakai saat preprocessing data ini
        (biasanya scaler.pkl dari data/models/).

    Returns
    -------
    dict dengan struktur sama seperti evaluate_all(): {"overall": ..., "per_feature": ...}
    tapi kali ini metriknya dihitung di SATUAN ASLI, bukan 0-1.
    """
    y_actual_scaled = np.asarray(y_actual_scaled, dtype=float)
    y_pred_scaled = np.asarray(y_pred_scaled, dtype=float)

    y_actual_orig = scaler.inverse_transform(y_actual_scaled)
    y_pred_orig = scaler.inverse_transform(np.clip(y_pred_scaled, 0, 1))

    return evaluate_all(y_actual_orig, y_pred_orig, feature_names, accuracy_method="mape")