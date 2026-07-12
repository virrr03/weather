# scripts/plot_evaluation.py

import os
import json
from datetime import datetime

import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = os.path.join(BASE_DIR, "data", "results")
LATEST_DIR = os.path.join(RESULT_DIR, "latest")
HISTORY_DIR = os.path.join(RESULT_DIR, "history")

os.makedirs(LATEST_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


def save_metrics(metrics, output_dir):
    save_path = os.path.join(output_dir, "evaluation_metrics.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Data evaluasi disimpan di {save_path}")

    return save_path


def plot_overall_evaluation(overall_metrics, output_dir):
    metric_names = list(overall_metrics.keys())
    metric_values = list(overall_metrics.values())

    plt.figure(figsize=(8, 5))
    plt.bar(metric_names, metric_values)
    plt.title("Evaluasi Gabungan Model RNN")
    plt.xlabel("Metrik Evaluasi")
    plt.ylabel("Nilai")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    save_path = os.path.join(output_dir, "overall_evaluation.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Grafik evaluasi gabungan disimpan di {save_path}")

    return save_path


def plot_feature_evaluation(per_feature_metrics, output_dir):
    features = list(per_feature_metrics.keys())

    mae_values = [per_feature_metrics[f]["MAE"] for f in features]
    rmse_values = [per_feature_metrics[f]["RMSE"] for f in features]
    mape_values = [per_feature_metrics[f]["MAPE"] for f in features]

    # =========================
    # MAE
    # =========================
    plt.figure(figsize=(10, 5))
    plt.bar(features, mae_values)
    plt.title("Evaluasi MAE Per Parameter Cuaca")
    plt.xlabel("Parameter Cuaca")
    plt.ylabel("MAE")
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    mae_path = os.path.join(output_dir, "feature_mae.png")
    plt.savefig(mae_path, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # RMSE
    # =========================
    plt.figure(figsize=(10, 5))
    plt.bar(features, rmse_values)
    plt.title("Evaluasi RMSE Per Parameter Cuaca")
    plt.xlabel("Parameter Cuaca")
    plt.ylabel("RMSE")
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    rmse_path = os.path.join(output_dir, "feature_rmse.png")
    plt.savefig(rmse_path, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # MAPE
    # =========================
    plt.figure(figsize=(10, 5))
    plt.bar(features, mape_values)
    plt.title("Evaluasi MAPE Per Parameter Cuaca")
    plt.xlabel("Parameter Cuaca")
    plt.ylabel("MAPE (%)")
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    mape_path = os.path.join(output_dir, "feature_mape.png")
    plt.savefig(mape_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Grafik MAE disimpan di {mae_path}")
    print(f"✅ Grafik RMSE disimpan di {rmse_path}")
    print(f"✅ Grafik MAPE disimpan di {mape_path}")

    return {
        "feature_mae": mae_path,
        "feature_rmse": rmse_path,
        "feature_mape": mape_path
    }


def save_all_outputs(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = save_metrics(metrics, output_dir)

    overall_path = plot_overall_evaluation(
        metrics["overall"],
        output_dir
    )

    feature_paths = plot_feature_evaluation(
        metrics["per_feature"],
        output_dir
    )

    return {
        "metrics_file": metrics_path,
        "overall_graph": overall_path,
        **feature_paths
    }


def plot_all_evaluation(metrics):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    history_output_dir = os.path.join(HISTORY_DIR, timestamp)

    latest_paths = save_all_outputs(metrics, LATEST_DIR)
    history_paths = save_all_outputs(metrics, history_output_dir)

    return {
        "latest": latest_paths,
        "history": history_paths,
        "timestamp": timestamp
    }