"""Inference: loads the saved weights and writes <ROLL>_pred_var<k>.csv (single column y)."""
import numpy as np
import pandas as pd
from polyreg import predict

ROLL = "BT2024251"

for var in (1, 2):
    X_test = pd.read_csv(f"{ROLL}/{ROLL}_test_var{var}.csv").to_numpy()
    m = dict(np.load(f"models/model_var{var}.npz"))
    pd.DataFrame({"y": predict(m, X_test)}).to_csv(f"{ROLL}_pred_var{var}.csv", index=False)
    print(f"wrote {ROLL}_pred_var{var}.csv ({len(X_test)} rows)")
