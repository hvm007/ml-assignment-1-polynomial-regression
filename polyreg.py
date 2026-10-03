"""
Polynomial regression from scratch (NumPy only).

Model:  y = w0 + sum_j w_j * phi_j(x)   where phi_j are all monomials of the inputs up to degree d.
The weights are fitted by
    OLS    minimise (1/N)  ||y - Phi w||^2
    Ridge  minimise (1/N)  ||y - Phi w||^2 + lam * ||w||_2^2      (L2)
    Lasso  minimise (1/2N) ||y - Phi w||^2 + lam * ||w||_1        (L1)
"""
from itertools import combinations_with_replacement
import numpy as np


# ---------------------------------------------------------------- features
def poly_powers(n_features, degree):
    """Exponent table: one row per monomial, one column per input feature.
    e.g. 2 features, degree 2 -> x1, x2, x1^2, x1*x2, x2^2
         rows: [1,0], [0,1], [2,0], [1,1], [0,2]
    Every row sums to at most `degree` (the definition used in the assignment)."""
    rows = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            p = [0] * n_features
            for i in combo:          # combo like (0, 0, 2) means x1 * x1 * x3
                p[i] += 1
            rows.append(p)
    return np.array(rows)


def poly_features(X, powers):
    """Build the design matrix Phi (N x n_terms) from raw inputs X (N x n_features)."""
    Phi = np.ones((X.shape[0], powers.shape[0]))
    for j in range(X.shape[1]):
        Phi *= X[:, [j]] ** powers[:, j]
    return Phi


# ---------------------------------------------------------------- solvers
# All solvers receive standardised features Z (zero mean, unit std) and centred y,
# so no intercept is needed inside them.
def fit_ols(Z, y):
    # least squares via SVD (stable even when Z'Z is ill-conditioned)
    return np.linalg.lstsq(Z, y, rcond=None)[0]


def fit_ridge(Z, y, lam):
    # closed form: (Z'Z/N + lam I) w = Z'y/N
    n, p = Z.shape
    return np.linalg.solve(Z.T @ Z / n + lam * np.eye(p), Z.T @ y / n)


def fit_lasso(Z, y, lam, w=None, max_sweeps=300, tol=1e-4):
    """Coordinate descent. Update one weight at a time, holding the others fixed:
           w_j = soft_threshold(rho_j, lam) / G_jj
       rho_j = correlation of feature j with the residual that ignores feature j.
       soft_threshold sets w_j to exactly 0 when |rho_j| <= lam -> sparse solution."""
    n, p = Z.shape
    G = Z.T @ Z / n            # Gram matrix, computed once
    c = Z.T @ y / n
    w = np.zeros(p) if w is None else w.copy()
    for _ in range(max_sweeps):
        biggest_change = 0.0
        for j in range(p):
            rho = c[j] - G[j] @ w + G[j, j] * w[j]
            new = np.sign(rho) * max(abs(rho) - lam, 0.0) / G[j, j]
            biggest_change = max(biggest_change, abs(new - w[j]))
            w[j] = new
        if biggest_change < tol:
            break
    return w


# ---------------------------------------------------------------- model wrapper
LAMBDAS = {
    "OLS": [0.0],
    "Ridge": list(np.logspace(-6, 1, 15)),
    "Lasso": list(np.logspace(-1, -3, 9)),   # large -> small so each fit warm-starts the next
}


def fit_path(X, y, degree, model):
    """Fit one model type for every lambda in its grid. Returns a list of fitted models."""
    powers = poly_powers(X.shape[1], degree)
    Phi = poly_features(X, powers)
    mean, std = Phi.mean(axis=0), Phi.std(axis=0)
    Z = (Phi - mean) / std
    y_mean = y.mean()
    yc = y - y_mean

    fitted, w = [], None
    for lam in LAMBDAS[model]:
        if model == "OLS":
            w = fit_ols(Z, yc)
        elif model == "Ridge":
            w = fit_ridge(Z, yc, lam)
        else:
            w = fit_lasso(Z, yc, lam, w)
        fitted.append({"powers": powers, "mean": mean, "std": std,
                       "w": w.copy(), "y_mean": y_mean, "lam": lam})
    return fitted


def predict(m, X):
    Z = (poly_features(X, m["powers"]) - m["mean"]) / m["std"]
    return Z @ m["w"] + m["y_mean"]


# ---------------------------------------------------------------- cross-validation
def kfold_indices(n, k=5, seed=0):
    """Shuffle the row indices and split them into k blocks."""
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, k)


def cv_mse(X, y, degree, model, k=5, seed=0):
    """K-fold CV. Returns the mean and the std-dev (across folds) of the validation MSE
    for every lambda in the grid."""
    folds = kfold_indices(len(y), k, seed)
    errors = np.zeros((k, len(LAMBDAS[model])))
    for i, val in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        for l, m in enumerate(fit_path(X[train], y[train], degree, model)):
            errors[i, l] = np.mean((y[val] - predict(m, X[val])) ** 2)
    return errors.mean(axis=0), errors.std(axis=0)
