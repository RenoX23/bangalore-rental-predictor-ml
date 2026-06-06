import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Bangalore Rent Predictor",
    page_icon="🏠",
    layout="wide"
)

# ── Load model + data ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load('models/xgb_model.pkl')
    features = joblib.load('models/features_list.pkl')
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv('data/features.csv')

model, FEATURES = load_model()
df = load_data()

# ── Precompute zone means for encoding ────────────────────────────────────
zone_means = {
    'West':    36703.85,
    'Central': 29882.35,
    'North':   26074.07,
    'South':   20297.22,
    'East':    19117.60,
}

# ── Sidebar nav ───────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigate", ["🏠 Rent Predictor", "📊 Model Performance", "🔍 SHAP Analysis"])

# ════════════════════════════════════════════════════════════════════════
# PAGE 1 — RENT PREDICTOR
# ════════════════════════════════════════════════════════════════════════
if page == "🏠 Rent Predictor":
    st.title("🏠 Bangalore Rent Predictor")
    st.caption("XGBoost model · Trained on 427 listings · MAPE 29.8%")

    col1, col2 = st.columns(2)

    with col1:
        bhk       = st.selectbox("BHK Type", [1, 2, 3, 4], index=1,
                                  format_func=lambda x: f"{x} BHK")
        furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"])
        zone       = st.selectbox("City Zone", list(zone_means.keys()))

    with col2:
        sqft           = st.slider("Area (sqft)", 200, 4000, 1000, step=50)
        locality_count = st.slider("Locality Demand (listings in area)", 1, 50, 10,
                                   help="Higher = more active rental market in that locality")

    furnish_map  = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
    zone_enc     = zone_means[zone]
    furnish_enc  = furnish_map[furnishing]
    log_sqft     = np.log1p(sqft)
    zone_x_furn  = zone_enc * furnish_enc

    input_data = pd.DataFrame([{
        'bhk_encoded':        bhk,
        'furnishing_encoded': furnish_enc,
        'log_sqft':           log_sqft,
        'locality_count':     locality_count,
        'zone_target_enc':    zone_enc,
        'zone_x_furnishing':  zone_x_furn,
    }])

    pred = model.predict(input_data)[0]
    low  = pred * 0.75
    high = pred * 1.25

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Rent",  f"₹{pred:,.0f}")
    m2.metric("Likely Range Low", f"₹{low:,.0f}")
    m3.metric("Likely Range High", f"₹{high:,.0f}")

    # SHAP waterfall for this prediction
    st.subheader("Why this prediction?")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[0],
            base_values   = explainer.expected_value,
            data          = input_data.iloc[0].values,
            feature_names = FEATURES
        ),
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    results = pd.DataFrame([
        {"Model": "Linear Regression", "RMSE": 20779, "MAE": 11999, "MAPE": 71.9, "R²": 0.203},
        {"Model": "Random Forest",     "RMSE": 13815, "MAE":  7079, "MAPE": 31.3, "R²": 0.643},
        {"Model": "XGBoost",           "RMSE": 14893, "MAE":  7081, "MAPE": 29.8, "R²": 0.617},
    ])

    st.subheader("Model Comparison")
    st.dataframe(results.set_index("Model"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(results, x="Model", y="RMSE", color="Model",
                     title="RMSE by Model (lower = better)",
                     color_discrete_sequence=["#ef5350","#42a5f5","#66bb6a"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(results, x="Model", y="MAPE", color="Model",
                      title="MAPE % by Model (lower = better)",
                      color_discrete_sequence=["#ef5350","#42a5f5","#66bb6a"])
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Prediction vs Actual (XGBoost)")
    preds = model.predict(df[FEATURES])
    fig3  = px.scatter(x=df['rent_monthly'], y=preds,
                       labels={"x": "Actual Rent (₹)", "y": "Predicted Rent (₹)"},
                       opacity=0.6, color_discrete_sequence=["#42a5f5"])
    max_val = max(df['rent_monthly'].max(), preds.max())
    fig3.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                              mode='lines', name='Perfect',
                              line=dict(color='red', dash='dash')))
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP ANALYSIS
# ════════════════════════════════════════════════════════════════════════
elif page == "🔍 SHAP Analysis":
    st.title("🔍 SHAP Feature Analysis")
    st.caption("What drives rental prices in Bangalore?")

    st.subheader("Global Feature Importance")
    shap_importance = pd.DataFrame({
        "Feature": FEATURES,
        "Mean |SHAP|": [11127.6, 1677.2, 1628.2, 1112.9, 1042.5, 362.9]
    }).sort_values("Mean |SHAP|", ascending=True)

    fig = px.bar(shap_importance, x="Mean |SHAP|", y="Feature",
                 orientation='h', color="Mean |SHAP|",
                 color_continuous_scale="Blues",
                 title="Mean absolute SHAP value per feature")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("SHAP Summary Plot")
    st.image('assets/shap_summary.png')

    st.subheader("Sample Waterfall Plots")
    col1, col2, col3 = st.columns(3)
    col1.image('assets/shap_waterfall_budget_1bhk.png',    caption="Budget 1BHK")
    col2.image('assets/shap_waterfall_mid_2bhk.png',       caption="Mid-range 2BHK")
    col3.image('assets/shap_waterfall_premium_3bhk.png',   caption="Premium 3BHK")

    st.subheader("Key Finding")
    st.info(
        "**Square footage is the dominant pricing driver** — SHAP importance of 11,127 vs "
        "1,677 for BHK type (6.6x gap). Zone × furnishing interaction captures "
        "the premium that furnished properties command specifically in high-rent zones. "
        "Furnishing status alone contributes minimally — it only matters in context of location."
    )
