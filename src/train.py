import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv('data/features.csv')

FEATURES = [
    'bhk_encoded', 'furnishing_encoded', 'log_sqft',
    'locality_count', 'zone_target_enc', 'zone_x_furnishing',
]
TARGET = 'rent_monthly'

X = df[FEATURES].values
y = df[TARGET].values

# ── Metrics helper ────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(name, model, X, y, cv):
    results = cross_validate(
        model, X, y, cv=cv,
        scoring=['neg_root_mean_squared_error',
                 'neg_mean_absolute_error',
                 'r2'],
        return_train_score=False
    )
    rmse = -results['test_neg_root_mean_squared_error'].mean()
    mae  = -results['test_neg_mean_absolute_error'].mean()
    r2   =  results['test_r2'].mean()

    # MAPE via manual fold
    mape_scores = []
    for train_idx, val_idx in cv.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        mape_scores.append(mape(y[val_idx], preds))
    mape_cv = np.mean(mape_scores)

    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"  RMSE : ₹{rmse:,.0f}")
    print(f"  MAE  : ₹{mae:,.0f}")
    print(f"  MAPE : {mape_cv:.1f}%")
    print(f"  R²   : {r2:.3f}")

    return {"model": name, "RMSE": rmse, "MAE": mae,
            "MAPE": mape_cv, "R2": r2}

# ── Models ────────────────────────────────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest",     RandomForestRegressor(n_estimators=200, random_state=42)),
    ("XGBoost",           XGBRegressor(n_estimators=300, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=42, verbosity=0)),
]

results = []
for name, model in models:
    res = evaluate_model(name, model, X, y, cv)
    results.append(res)

# ── Comparison table ──────────────────────────────────────────────────────
print("\n\n── Model Comparison ──────────────────────────────────────")
results_df = pd.DataFrame(results).set_index("model")
results_df["RMSE"] = results_df["RMSE"].apply(lambda x: f"₹{x:,.0f}")
results_df["MAE"]  = results_df["MAE"].apply(lambda x: f"₹{x:,.0f}")
results_df["MAPE"] = results_df["MAPE"].apply(lambda x: f"{x:.1f}%")
results_df["R2"]   = results_df["R2"].apply(lambda x: f"{x:.3f}")
print(results_df.to_string())

# ── Train final XGBoost on full data + serialize ──────────────────────────
print("\nTraining final XGBoost on full dataset...")
final_model = XGBRegressor(n_estimators=300, max_depth=4,
                            learning_rate=0.05, subsample=0.8,
                            random_state=42, verbosity=0)
final_model.fit(X, y)
joblib.dump(final_model, 'models/xgb_model.pkl')
joblib.dump(FEATURES, 'models/features_list.pkl')
print("Saved → models/xgb_model.pkl")
