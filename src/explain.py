import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Load model + data ─────────────────────────────────────────────────────
model    = joblib.load('models/xgb_model.pkl')
FEATURES = joblib.load('models/features_list.pkl')

df = pd.read_csv('data/features.csv')
X  = df[FEATURES]
y  = df['rent_monthly']

# ── SHAP explainer ────────────────────────────────────────────────────────
print("Computing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
print("Done.")

# ── Plot 1: Summary plot ──────────────────────────────────────────────────
plt.figure()
shap.summary_plot(shap_values, X, feature_names=FEATURES, show=False)
plt.title("SHAP Summary — Feature Impact on Rent Prediction")
plt.tight_layout()
plt.savefig('assets/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved → assets/shap_summary.png")

# ── Plot 2: Waterfall for 3 sample properties ─────────────────────────────
samples = {
    "budget_1bhk":    df[(df['bhk_encoded'] == 1) & (df['furnishing_encoded'] == 0)].index[0],
    "mid_2bhk":       df[(df['bhk_encoded'] == 2) & (df['furnishing_encoded'] == 1)].index[0],
    "premium_3bhk":   df[(df['bhk_encoded'] == 3) & (df['furnishing_encoded'] == 2)].index[0],
}

for label, idx in samples.items():
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[idx],
            base_values   = explainer.expected_value,
            data          = X.iloc[idx].values,
            feature_names = FEATURES
        ),
        show=False
    )
    actual = y.iloc[idx]
    pred   = model.predict(X.iloc[[idx]])[0]
    plt.title(f"{label} | Actual: ₹{actual:,.0f} | Predicted: ₹{pred:,.0f}")
    plt.tight_layout()
    plt.savefig(f'assets/shap_waterfall_{label}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → assets/shap_waterfall_{label}.png")

# ── Plot 3: Dependence plot — top feature ─────────────────────────────────
top_feature = FEATURES[np.abs(shap_values).mean(0).argmax()]
print(f"\nTop SHAP feature: {top_feature}")

plt.figure(figsize=(8, 5))
shap.dependence_plot(top_feature, shap_values, X,
                     feature_names=FEATURES, show=False)
plt.title(f"SHAP Dependence — {top_feature} vs Rent Impact")
plt.tight_layout()
plt.savefig('assets/shap_dependence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved → assets/shap_dependence.png")

# ── Print global feature importance ──────────────────────────────────────
mean_shap = pd.Series(
    np.abs(shap_values).mean(0),
    index=FEATURES
).sort_values(ascending=False)

print("\nGlobal SHAP importance:")
print(mean_shap)
