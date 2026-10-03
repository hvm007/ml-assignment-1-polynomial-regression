"""
Compares a restricted feature subset against using all features.
   var1: first 3 features (x1,x2,x3)  vs  all 6
   var2: first feature only (x1)      vs  all 3
Run after train.py (it reads results/cv_var<k>.csv for the 'all features' curves).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polyreg import cv_mse

ROLL = "BT2024251"
HINT = {1: (3, 3), 2: (1, 4)}        # var -> (number of leading features, degree to highlight)

for var, (k, hint_degree) in HINT.items():
    df = pd.read_csv(f"{ROLL}/{ROLL}_train_var{var}.csv")
    X, y = df.drop(columns="y").to_numpy(), df["y"].to_numpy()
    full = pd.read_csv(f"results/cv_var{var}.csv")
    degrees = sorted(full.degree.unique())

    rows = []
    for d in degrees:
        mse, std = cv_mse(X[:, :k], y, d, "OLS")
        rows.append({"degree": d, "cv_mse": mse[0], "cv_std": std[0], "cv_r2": 1 - mse[0] / y.var()})
    hint = pd.DataFrame(rows)
    hint.to_csv(f"results/hint_var{var}.csv", index=False)

    ols = full[full.model == "OLS"]
    best = full.loc[full.groupby("degree").cv_mse.idxmin()]     # best of OLS/Ridge/Lasso per degree
    label = "x1 only" if k == 1 else f"first {k} features"
    plt.plot(hint.degree, hint.cv_mse, marker="s", color="tab:red", label=f"{label} (OLS)")
    plt.plot(ols.degree, ols.cv_mse, marker="o", color="tab:orange", label="all features (OLS)")
    plt.plot(best.degree, best.cv_mse, marker="o", color="tab:blue", label="all features (best of OLS/Ridge/Lasso)")
    h = hint[hint.degree == hint_degree].iloc[0]
    plt.scatter([hint_degree], [h.cv_mse], s=220, facecolors="none", edgecolors="k", zorder=5,
                label=f"{label}, degree {hint_degree} (MSE {h.cv_mse:.2f})")
    plt.yscale("log"); plt.xlabel("polynomial degree"); plt.ylabel("7-fold CV MSE (log scale)")
    plt.title(f"var{var}: {label} vs all features")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.savefig(f"results/hint_var{var}.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"var{var}: restricted ({label}, degree {hint_degree}) CV MSE={h.cv_mse:.3f} R2={h.cv_r2:.3f} "
          f"| chosen model CV MSE={full.cv_mse.min():.3f}")
