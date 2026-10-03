"""
Writes the fitted weights in readable form: results/weights_var<k>.csv, one row per polynomial term.
  weight_standardised : weight on the standardised term (comparable across terms; this is what is fitted)
  coefficient_raw     : coefficient on the raw term, so that  y = intercept + sum(coefficient_raw * term)
Run after train.py.
"""
import numpy as np
import pandas as pd


def term_name(p):
    parts = [f"x{j + 1}" if e == 1 else f"x{j + 1}^{e}" for j, e in enumerate(p) if e > 0]
    return "*".join(parts)


for var in (1, 2):
    m = np.load(f"models/model_var{var}.npz")
    w, mean, std, powers = m["w"], m["mean"], m["std"], m["powers"]
    raw = w / std
    intercept = float(m["y_mean"]) - np.sum(raw * mean)

    df = pd.DataFrame({"term": [term_name(p) for p in powers],
                       "degree": powers.sum(axis=1),
                       "weight_standardised": w,
                       "coefficient_raw": raw})
    df = df.reindex(df.weight_standardised.abs().sort_values(ascending=False).index)
    top = pd.DataFrame([{"term": "intercept", "degree": 0,
                         "weight_standardised": float(m["y_mean"]), "coefficient_raw": intercept}])
    pd.concat([top, df]).to_csv(f"results/weights_var{var}.csv", index=False)

    nz = int((df.weight_standardised.abs() > 1e-8).sum())
    print(f"var{var}: degree {int(m['degree'])} {m['model']}, {nz} of {len(df)} weights non-zero")
    print(df.head(8).round(4).to_string(index=False), "\n")
