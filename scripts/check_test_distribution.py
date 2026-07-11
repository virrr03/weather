import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_PATH = os.path.join(BASE_DIR, "data", "processed", "anfis_split.npz")

data = np.load(SPLIT_PATH)
y_test_orig = data["y_test_orig"]

NAMA_VARIABEL = ["Temperature", "Humidity", "Pressure", "WindSpeed", "Irradiance"]

for i, n in enumerate(NAMA_VARIABEL):
    col = y_test_orig[:, i]
    print(f"{n}: min={col.min():.3f} max={col.max():.3f} std={col.std():.3f} n_unique={len(np.unique(col))}")