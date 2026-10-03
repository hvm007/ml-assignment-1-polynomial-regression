"""
Feature-subset study: for every non-empty subset of the input features, sweep the polynomial
degree with least squares and Ridge (7-fold CV) and record the best setting for that subset.
   var1: 63 subsets of 6 features      var2: 7 subsets of 3 features
Usage: python feature_subsets.py [1|2]
Lasso is left out here to keep the run short; least squares and Ridge are both closed-form.
"""
import sys
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polyreg import LAMBDAS, cv_mse

ROLL = "BT2024251"
MAX_DEGREE = {1: 6, 2: 12}


def study(var):
    df = pd.read_csv(f"{ROLL}/{ROLL}_train_var{var}.csv")
    X, y = df.drop(columns="y").to_numpy(), df["y"].to_numpy()
    D = X.shape[1]

    rows = []
    for size in range(1, D + 1):
        for subset in combinations(range(D), size):
            best = None
            for degree in range(1, MAX_DEGREE[var] + 1):
                for model in ("Least squares", "Ridge"):
                    mse, std = cv_mse(X[:, subset], y, degree, model)
                    i = int(np.argmin(mse))
                    if best is None or mse[i] < best["cv_mse"]:
                        best = {"features": " ".join(f"x{j + 1}" for j in subset), "n_features": size,
                                "degree": degree, "model": model, "lam": LAMBDAS[model][i],
                                "cv_mse": mse[i], "cv_std": std[i], "cv_r2": 1 - mse[i] / y.var()}
            rows.append(best)
            print(f"var{var}  {best['features']:18s} best: degree {best['degree']:2d} {best["model"]:13s} "
                  f"CV MSE={best['cv_mse']:.3f}", flush=True)
    res = pd.DataFrame(rows).sort_values("cv_mse")
    res.to_csv(f"results/subsets_var{var}.csv", index=False)

    plt.scatter(res.n_features, res.cv_mse, s=30, alpha=0.7)
    if D <= 3:                                   # few enough points to label each one
        for _, r in res.iterrows():
            plt.annotate(r.features, (r.n_features, r.cv_mse), textcoords="offset points",
                         xytext=(7, -3), fontsize=8)
    plt.yscale("log"); plt.xticks(range(1, D + 1))
    plt.xlabel("number of input features used"); plt.ylabel("best 7-fold CV MSE (log scale)")
    plt.title(f"var{var}: every feature subset, best degree for each")
    plt.grid(alpha=0.3)
    plt.savefig(f"results/subsets_var{var}.png", dpi=150, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    for v in ([int(sys.argv[1])] if len(sys.argv) > 1 else [1, 2]):
        study(v)
