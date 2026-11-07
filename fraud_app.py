import os
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


#Page layout configuration
st.set_page_config(page_title="Fintech Fraud Risk Engine", layout='wide', page_icon="🛡️")

#Header
st.title("🛡️ Live Fintech Transaction Risk Engine & Explainable AI")
st.markdown("Real-time transaction fraud scoring with XGBoost and SHAP explainability")



st.sidebar.header("Live Transaction Telemetry")

txn_amount = st.sidebar.number_input(label="Transaction Input ($)", min_value=1.0, max_value=10000.0, value=350.0, step=25.0)

distance_from_home_km = st.sidebar.slider(label="Distance from Home (KM)", min_value=1.0, max_value=1000.0, value=1.0, step=1.0)

failed_pin_attempts = st.sidebar.slider(label="Failed Pin Attempts", min_value=0, max_value=5, value=0, step=1)

customer_tenure_months = st.sidebar.slider(label="Customer Tenure (Months)", min_value=1, max_value=60, value=1, step=1)

hour_of_day = st.sidebar.slider(label='Hour of day (0-23)', min_value=0, max_value=23, value=14, step=1)

device_type = st.sidebar.selectbox(label="Device Type", options=["Android", "iOS", "Web"])

is_international = 1 if st.sidebar.checkbox(
label="International Transaction (Cross-Border)"
) else 0


@st.cache_resource
def load_artifacts():
    if os.path.exists('fraud_detection_pipeline.joblib'):
        actual_path = 'fraud_detection_pipeline.joblib'
    elif os.path.exists('Exercises/fraud_detection_pipeline.joblib'):
        actual_path = 'Exercises/fraud_detection_pipeline.joblib'
    else:
        raise FileNotFoundError("Model file fraud_detection_pipeline.joblib not found")

    bundle = joblib.load(actual_path)

    explainer = shap.TreeExplainer(bundle['model'])

    return bundle, explainer

bundle, explainer = load_artifacts()

input_raw = pd.DataFrame([{
    'customer_tenure_months': customer_tenure_months,
    'txn_amount': txn_amount,
    'distance_from_home_km': distance_from_home_km,
    'hour_of_day': hour_of_day,
    'is_international': is_international,
    'failed_pin_attempts': failed_pin_attempts,
    'device_type': device_type
}])

trans = bundle['preprocessor'].transform(input_raw)
trans_df = pd.DataFrame(trans, columns=bundle['encoded_features'])
prob = bundle['model'].predict_proba(trans_df)[0,1]
threshold = bundle.get('threshold', 0.30)

st.subheader("📊 Real-Time Risk Assessment")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label='Fraud Risk Probability', value=f"{prob * 100:.1f}%")

with col2:
    if prob >= threshold:
        st.error("🚨 FLAGGED FOR REVIEW (High Risk)")
    else:
        st.success("✅ TRANSACTION APPROVED (Clean)")
with col3:
    st.metric(label="Operating Risk Threshold", value=f"{threshold * 100:.0f}%")


st.subheader("🔍 Local Prediction Forensics: Why Did the Model Assign This Score")
shap_values = explainer(trans_df)
fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
shap.plots.waterfall(shap_values[0], show=False)
plt.title(f"SHAP Waterfall Attribution: Fraud Probability = {prob*100:.1f}%", fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
