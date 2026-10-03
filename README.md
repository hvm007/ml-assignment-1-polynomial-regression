# ML Assignment 1 - Polynomial Regression (BT2024251)

Polynomial regression written from scratch in NumPy.

    pip install -r requirements.txt
    python train.py      # 7-fold CV over degree, model and lambda; saves models/ and results/
    python predict.py    # writes BT2024251_pred_var1.csv and BT2024251_pred_var2.csv

- `polyreg.py` - polynomial features, OLS, Ridge (closed form), Lasso (coordinate descent), K-fold CV
- `train.py` - model selection and final fit
- `predict.py` - inference on the test files
- `feature_subsets.py` - degree sweep for every subset of the input features (63 for var1, 7 for var2)
- `export_weights.py` - writes the fitted weights per term to `results/weights_var<k>.csv` (run after train.py)
- `results/` - CV tables, CV-MSE-vs-degree plots, predicted-vs-actual plots
