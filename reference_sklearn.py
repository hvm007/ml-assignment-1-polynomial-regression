"""
scikit-learn reference, used only to cross-check the NumPy implementation in train.py.
Usage: python reference_sklearn.py            (trains var1 and var2)
       python reference_sklearn.py 1          (only var1)

For every problem we try polynomial degrees 1..D with three linear models
(plain least squares, Ridge, Lasso), score each with 5-fold cross-validation,
and keep the simplest setting whose CV MSE is within 2% of the best one.
"""
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

ROLL = "BT2024251"
MAX_DEGREE = {1: 6, 2: 12}      # degree-7 on 6 features = 1716 terms > 1000 rows, so we stop at 6
SEED = 0


def make_model(name, degree):
    poly = PolynomialFeatures(degree, include_bias=False)
    if name == "OLS":
        reg = LinearRegression()
    elif name == "Ridge":
        reg = RidgeCV(alphas=np.logspace(-6, 2, 17))
    else:
        reg = LassoCV(cv=5, n_alphas=30, max_iter=20000, n_jobs=-1)
    return make_pipeline(poly, StandardScaler(), reg)


def train(var):
    df = pd.read_csv(f"{ROLL}/{ROLL}_train_var{var}.csv")
    X, y = df.drop(columns="y"), df["y"]
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for degree in range(1, MAX_DEGREE[var] + 1):
        for name in ("OLS", "Ridge", "Lasso"):
            mse = -cross_val_score(make_model(name, degree), X, y, cv=cv,
                                   scoring="neg_mean_squared_error").mean()
            rows.append({"degree": degree, "model": name, "cv_mse": mse,
                         "cv_r2": 1 - mse / y.var(ddof=0)})
            print(f"var{var}  degree={degree:2d}  {name:5s}  CV MSE={mse:.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(f"results/sklearn_cv_var{var}.csv", index=False)

    # simplest (lowest degree) setting within 2% of the best CV MSE
    good = res[res.cv_mse <= 1.02 * res.cv_mse.min()]
    best = good.sort_values(["degree", "cv_mse"]).iloc[0]
    print(f"--> var{var}: chose degree {best.degree} with {best.model} "
          f"(CV MSE={best.cv_mse:.4f}, R2={best.cv_r2:.4f})\n")

    # plot CV MSE vs degree
    for name, g in res.groupby("model"):
        plt.plot(g.degree, g.cv_mse, marker="o", label=name)
    plt.yscale("log")
    plt.xlabel("polynomial degree")
    plt.ylabel("5-fold CV MSE (log scale)")
    plt.title(f"var{var}: model selection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"results/sklearn_cv_var{var}.png", dpi=150, bbox_inches="tight")
    plt.close()




if __name__ == "__main__":
    for v in ([int(sys.argv[1])] if len(sys.argv) > 1 else [1, 2]):
        train(v)
