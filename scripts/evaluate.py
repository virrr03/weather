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
    Karena data sudah dinormalisasi 0-1,
    akurasi dihitung dari MAE.

    Accuracy = (1 - MAE) * 100
    """
    accuracy = (1 - mae) * 100

    # Batasi agar tidak minus dan tidak lebih dari 100
    accuracy = max(0, min(accuracy, 100))

    return round(float(accuracy), 2)


def evaluate_overall(y_actual, y_pred):
    """
    Evaluasi gabungan seluruh parameter cuaca.
    """
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mape = safe_mape(y_actual, y_pred)

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


def evaluate_per_feature(y_actual, y_pred, feature_names):
    """
    Evaluasi tiap parameter cuaca.
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


def evaluate_all(y_actual, y_pred, feature_names):
    """
    Menghasilkan evaluasi gabungan dan evaluasi per parameter.
    """
    return {
        "overall": evaluate_overall(y_actual, y_pred),
        "per_feature": evaluate_per_feature(y_actual, y_pred, feature_names)
    }