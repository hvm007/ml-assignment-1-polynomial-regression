"""
Training (NumPy only).   python train.py        -> both problems
                          python train.py 1      -> only var1

For every degree and every model (least squares / Ridge / Lasso) we run 7-fold CV, keep the best
lambda, and finally choose the lowest degree whose CV MSE is within 2% of the best.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polyreg import LAMBDAS, cv_mse, fit_path, kfold_indices, predict

ROLL = "BT2024251"
MAX_DEGREE = {1: 6, 2: 12}   # var1: degree 7 on 6 features = 1715 terms > 1000 rows, so stop at 6


def train(var):
    df = pd.read_csv(f"{ROLL}/{ROLL}_train_var{var}.csv")
    X, y = df.drop(columns="y").to_numpy(), df["y"].to_numpy()

    rows = []
    for degree in range(1, MAX_DEGREE[var] + 1):
        for model in ("Least squares", "Ridge", "Lasso"):
            mse, std = cv_mse(X, y, degree, model)
            best = int(np.argmin(mse))
            rows.append({"degree": degree, "model": model, "lam": LAMBDAS[model][best],
                         "cv_mse": mse[best], "cv_std": std[best], "cv_r2": 1 - mse[best] / y.var()})
            print(f"var{var}  degree={degree:2d}  {model:13s}  lam={LAMBDAS[model][best]:.2e}  "
                  f"CV MSE={mse[best]:.4f} +- {std[best]:.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"results/cv_var{var}.csv", index=False)

    # lowest degree within 2% of the best CV MSE
    good = res[res.cv_mse <= 1.02 * res.cv_mse.min()]
    best = good.sort_values(["degree", "cv_mse"]).iloc[0]
    degree, model, lam = int(best.degree), best.model, best.lam
    print(f"--> var{var}: degree {degree}, {model}, lam={lam:.2e}, "
          f"CV MSE={best.cv_mse:.4f}, CV R2={best.cv_r2:.4f}\n")
    pick = LAMBDAS[model].index(lam)

    # plot 1: CV MSE vs degree
    for name, g in res.groupby("model"):
        plt.plot(g.degree, g.cv_mse, marker="o", label=name)
    plt.yscale("log"); plt.xlabel("polynomial degree"); plt.ylabel("7-fold CV MSE (log scale)")
    plt.title(f"var{var}: model selection"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"results/cv_var{var}.png", dpi=150, bbox_inches="tight"); plt.close()

    # plot 2: predicted vs actual, using only held-out predictions
    oof = np.zeros_like(y)
    folds = kfold_indices(len(y))
    for i, val in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
        oof[val] = predict(fit_path(X[tr], y[tr], degree, model)[pick], X[val])
    plt.scatter(y, oof, s=8, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=1)
    plt.xlabel("actual y"); plt.ylabel("predicted y (held-out folds)")
    plt.title(f"var{var}: degree {degree} {model}"); plt.grid(alpha=0.3)
    plt.savefig(f"results/pred_vs_actual_var{var}.png", dpi=150, bbox_inches="tight"); plt.close()

    # final fit on all 1000 rows, save the weights
    m = fit_path(X, y, degree, model)[pick]
    np.savez(f"models/model_var{var}.npz", **m, degree=degree, model=model)
    print(f"    non-zero weights: {int(np.sum(np.abs(m['w']) > 1e-8))} of {len(m['w'])}")


if __name__ == "__main__":
    for v in ([int(sys.argv[1])] if len(sys.argv) > 1 else [1, 2]):
        train(v)
