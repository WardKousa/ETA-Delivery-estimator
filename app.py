# app.py
import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Page config and scale fix
# -------------------------
st.set_page_config(
    page_title="ETA Estimator",
    layout="wide",  # full width
    initial_sidebar_state="expanded"
)

# -------------------------
# Load artifacts
# -------------------------
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(__file__)
    lr_path = os.path.join(BASE_DIR, "models", "linear.pkl")
    rf_path = os.path.join(BASE_DIR, "models", "rf.pkl")
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
    data_path = os.path.join(BASE_DIR, "data", "synthetic.csv")

    for path in [lr_path, rf_path, metrics_path, data_path]:
        if not os.path.exists(path):
            st.error(f"Required file not found: {path}")
            st.stop()

    lr = joblib.load(lr_path)
    rf = joblib.load(rf_path)

    with open(metrics_path) as f:
        metrics = json.load(f)

    df = pd.read_csv(data_path)

    return lr, rf, metrics, df

lr, rf, metrics, df = load_artifacts()

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Input features (simulate)")
distance = st.sidebar.slider("Distance (km)", min_value=1.0, max_value=200.0, value=20.0, step=1.0)
num_stops = st.sidebar.slider("Number of stops", min_value=0, max_value=20, value=5, step=1)
pickup_hour = st.sidebar.slider("Pickup hour (0-23)", min_value=0, max_value=23, value=10, step=1)
is_rush = st.sidebar.checkbox("Rush hour", value=False)
rain = st.sidebar.checkbox("Rain", value=False)
hub_dwell = st.sidebar.slider("Hub dwell (min)", min_value=0.0, max_value=120.0, value=10.0, step=1.0)

X = pd.DataFrame([{
    "distance_km": distance,
    "num_stops": num_stops,
    "pickup_hour": pickup_hour,
    "is_rush": int(is_rush),
    "rain": int(rain),
    "hub_dwell": hub_dwell
}])

# -------------------------
# Page Navigation
# -------------------------
page = st.sidebar.radio("Select page:", ["Home", "Linear Regression", "Random Forest", "Comparison", "Metrics Info"])

# -------------------------
# Home Page
# -------------------------
if page == "Home":
    st.markdown("<h1 style='font-size:80px;'>Delivery ETA Estimator — Prototype</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:16px;'>"
        "This app predicts delivery ETA using Linear Regression and Random Forest models trained on synthetic delivery data. "
        "You can adjust features in the sidebar and view the updated ETA predictions. "
        "Additionally, you can explore informative metrics, graphs, and benchmarks for each model, compare the models, "
        "and see a separate page explaining what each metric means, including residuals and top contributing features."
        "</p>",
        unsafe_allow_html=True
    )

    # Add vertical space
    st.markdown("<br>", unsafe_allow_html=True)  # two line breaks


    # ETA predictions
    col1, col_spacer, col2 = st.columns([1, 2, 1])

    pred_lr = lr.predict(X)[0]
    pred_rf = rf.predict(X)[0]

    # Label on top, value below
    col1.markdown(f"""
    <div style="text-align:center;">
        <div style="font-size:30px; font-weight:bold;">Linear Regression ETA (min)</div>
        <div style="font-size:45px; font-weight:normal; margin-top:5px;">{pred_lr:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div style="text-align:center;">
        <div style="font-size:30px; font-weight:bold;">Random Forest ETA (min)</div>
        <div style="font-size:45px; font-weight:normal; margin-top:5px;">{pred_rf:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Add vertical space
    st.markdown("<br><br>", unsafe_allow_html=True)  # one line breaks

    st.markdown("<h4 style='font-weight: bold;'>Model test MAE (minutes):</h4>", unsafe_allow_html=True)
    st.markdown(
        f"<h6 style='font-weight: normal;'>{metrics['mae_lr']:.2f} (Linear Regression), {metrics['mae_rf']:.2f} (Random Forest)</h6>",
        unsafe_allow_html=True)


# -------------------------
# Helper: Linear Regression
# -------------------------
def linear_regression_page():
    st.title("Linear Regression Analysis")
    pred_lr = lr.predict(X)[0]
    st.metric("ETA Prediction (min)", f"{pred_lr:.1f}")

    # Breakdown popup
    if st.button("Show ETA Breakdown (Linear Regression)"):
        st.write("ETA breakdown (feature * coefficient):")
        coefs = lr.coef_
        breakdown = {col: X[col][0]*coef for col, coef in zip(X.columns, coefs)}
        st.json(breakdown)

    # Global importance = coefficients
    st.subheader("Global Feature Importance (coefficients)")
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": lr.coef_
    }).sort_values("coefficient", key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(6,3))  # smaller bar chart
    coef_df.plot.bar(x="feature", y="coefficient", ax=ax, legend=False)
    st.pyplot(fig)

    # Top contributing features (delta from median)
    st.subheader("Top contributing features (delta from median)")
    median = df.median()
    base_pred = lr.predict(X)[0]
    contribs = {}
    for col in X.columns:
        X_mod = X.copy()
        X_mod[col] = X[col].dtype.type(median[col])
        contribs[col] = base_pred - float(lr.predict(X_mod)[0])
    contrib_df = pd.DataFrame(list(contribs.items()), columns=["feature","delta"])
    contrib_df["abs_delta"] = contrib_df["delta"].abs()
    contrib_df = contrib_df.sort_values("abs_delta", ascending=False)
    for _, row in contrib_df.head(3).iterrows():
        st.write(f"{row['feature']}: {row['delta']:.1f} minutes (delta from median)")

    # Residual diagnostics
    st.subheader("Residual diagnostics")
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    X_sample = sample_df.drop(columns=["eta_min"])
    y_true = sample_df["eta_min"].values
    y_pred = lr.predict(X_sample)
    residuals = y_pred - y_true

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6,3))
        ax1.scatter(y_pred, y_true, alpha=0.4, s=8)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=1)
        ax1.set_xlabel("Predicted ETA")
        ax1.set_ylabel("Actual ETA")
        ax1.set_title("Predicted vs Actual")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.scatter(sample_df["distance_km"], residuals, alpha=0.5, s=8)
        ax2.axhline(0, color='grey', linewidth=1)
        ax2.set_xlabel("Distance")
        ax2.set_ylabel("Residual")
        ax2.set_title("Residuals vs Distance")
        st.pyplot(fig2)

    st.write("Residual stats:")
    st.json({
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "median_residual": float(np.median(residuals))
    })

# -------------------------
# Helper: Random Forest
# -------------------------
def random_forest_page():
    st.title("Random Forest Analysis")
    pred_rf = rf.predict(X)[0]
    st.metric("ETA Prediction (min)", f"{pred_rf:.1f}")

    # Breakdown dropdown explanation
    with st.expander("Show ETA Breakdown (Random Forest)"):
        st.write(
            "Random Forest does not provide a feature-wise breakdown like Linear Regression because it is a non-linear "
            "ensemble model. Predictions are based on averaging multiple decision trees, making it impossible to attribute "
            "specific contributions to individual features."
        )

    # Global importance = permutation importance
    st.subheader("Global Feature Importance (permutation importance)")
    imp = metrics["importances"]
    imp_df = pd.DataFrame(list(imp.items()), columns=["feature", "importance"]).sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(6,3))  # smaller bar chart
    imp_df.plot.bar(x="feature", y="importance", ax=ax, legend=False)
    st.pyplot(fig)

    # Top contributing features (delta from median)
    st.subheader("Top contributing features (delta from median)")
    base_pred_rf = rf.predict(X)[0]
    contribs_rf = {}
    median = df.median()
    for col in X.columns:
        X_mod = X.copy()
        X_mod[col] = X[col].dtype.type(median[col])
        contribs_rf[col] = base_pred_rf - float(rf.predict(X_mod)[0])
    contrib_df_rf = pd.DataFrame(list(contribs_rf.items()), columns=["feature","delta"])
    contrib_df_rf["abs_delta"] = contrib_df_rf["delta"].abs()
    contrib_df_rf = contrib_df_rf.sort_values("abs_delta", ascending=False)
    for _, row in contrib_df_rf.head(3).iterrows():
        st.write(f"{row['feature']}: {row['delta']:.1f} minutes (delta from median)")

    # Residual diagnostics
    st.subheader("Residual diagnostics")
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    X_sample = sample_df.drop(columns=["eta_min"])
    y_true = sample_df["eta_min"].values
    y_pred = rf.predict(X_sample)
    residuals = y_pred - y_true

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6,3))
        ax1.scatter(y_pred, y_true, alpha=0.4, s=8)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=1)
        ax1.set_xlabel("Predicted ETA")
        ax1.set_ylabel("Actual ETA")
        ax1.set_title("Predicted vs Actual")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.scatter(sample_df["distance_km"], residuals, alpha=0.5, s=8)
        ax2.axhline(0, color='grey', linewidth=1)
        ax2.set_xlabel("Distance")
        ax2.set_ylabel("Residual")
        ax2.set_title("Residuals vs Distance")
        st.pyplot(fig2)

    st.write("Residual stats:")
    st.json({
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "median_residual": float(np.median(residuals))
    })

# -------------------------
# Helper: Comparison Page
# -------------------------
def comparison_page():
    st.title("Model Comparison")

    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    X_sample = sample_df.drop(columns=["eta_min"])
    y_true = sample_df["eta_min"].values

    # Residual stats
    y_pred_lr = lr.predict(X_sample)
    residuals_lr = y_pred_lr - y_true
    y_pred_rf = rf.predict(X_sample)
    residuals_rf = y_pred_rf - y_true

    # Global Feature Importance
    coef_dict = dict(zip(X.columns, lr.coef_))
    imp_dict = metrics["importances"]

    comp_data = {
        "Metric": ["MAE", "Mean Residual", "Std Residual", "Median Residual"] + [f"GFI {f}" for f in X.columns],
        "Linear Regression": [metrics["mae_lr"], np.mean(residuals_lr), np.std(residuals_lr), np.median(residuals_lr)] + [coef_dict[f] for f in X.columns],
        "Random Forest": [metrics["mae_rf"], np.mean(residuals_rf), np.std(residuals_rf), np.median(residuals_rf)] + [imp_dict[f] for f in X.columns]
    }

    comp_table = pd.DataFrame(comp_data)
    st.subheader("Comparison Table")
    st.dataframe(comp_table)

    st.subheader("Linear Regression Details")
    linear_regression_page()

    st.subheader("Random Forest Details")
    random_forest_page()

# -------------------------
# Metrics Info Page
# -------------------------
def metrics_info_page():
    st.title("Metrics Explained")
    st.write("""
    - **Models**: Linear Regression (simple additive coefficients) and Random Forest (ensemble of decision trees).
    - **Linear Regression Model**: predicts ETA as a weighted sum of features (ETA = sum(feature * coefficient)).
    - **Random Forest Model**: ensemble of decision trees; non-linear, does not give per-feature contributions.
    - **ETA**: Estimated Time of Arrival — predicted delivery time in minutes.
    - **MAE**: Mean Absolute Error — average absolute difference between predicted ETA and actual ETA. More info: [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error)
    - **Global Feature Importance (GFI)**: Shows how much each feature contributes to predictions. For Linear Regression, it's the coefficient; for Random Forest, it's computed via permutation importance.
    - **Top Contributing Features (delta from median)**: Shows which features contributed most to a specific prediction.
    - **Residuals**: Difference bAetween predicted ETA and actual ETA; used to assess prediction accuracy.
    """)
    st.write("Reference video explaining metrics and models:")
    st.write("[Metrics and Model Explanation - YouTube](https://www.youtube.com/watch?v=0Lt9w-BxKFQ)")

# -------------------------
# Page Rendering
# -------------------------
if page == "Linear Regression":
    linear_regression_page()
elif page == "Random Forest":
    random_forest_page()
elif page == "Comparison":
    comparison_page()
elif page == "Metrics Info":
    metrics_info_page()
