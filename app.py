import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Heart Disease Risk Dashboard", layout="wide")
st.title("Heart Disease Risk Dashboard")

model = joblib.load("heart_rf_pipeline.pkl")
sample_patient = joblib.load("sample_patient.pkl")
top_features = joblib.load("top_features.pkl")

st.subheader("Patient Input Form")
st.caption("Use the default values or edit manually, then click Predict.")

field_specs = {
    "age": ("Age", 20.0, 80.0, 1.0),
    "sex": ("Sex (0=female, 1=male)", 0.0, 1.0, 1.0),
    "cp": ("Chest Pain Type (0-3)", 0.0, 3.0, 1.0),
    "trestbps": ("Resting Blood Pressure (90-200)", 90.0, 200.0, 1.0),
    "chol": ("Cholesterol (100-600)", 100.0, 600.0, 1.0),
    "fbs": ("Fasting Blood Sugar >120 (0/1)", 0.0, 1.0, 1.0),
    "restecg": ("Resting ECG (0-2)", 0.0, 2.0, 1.0),
    "thalach": ("Max Heart Rate (70-210)", 70.0, 210.0, 1.0),
    "exang": ("Exercise Angina (0/1)", 0.0, 1.0, 1.0),
    "oldpeak": ("ST Depression (0-7)", 0.0, 7.0, 0.1),
    "slope": ("Slope (0-2)", 0.0, 2.0, 1.0),
    "ca": ("Major Vessels (0-3)", 0.0, 3.0, 1.0),
    "thal": ("Thal (1-3)", 1.0, 3.0, 1.0),
}

inputs = {}
cols = st.columns(2)
for i, (feature, (label, low, high, step)) in enumerate(field_specs.items()):
    default_val = float(sample_patient.get(feature, low))
    with cols[i % 2]:
        inputs[feature] = st.number_input(label, min_value=low, max_value=high, value=default_val, step=step)

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    pred = int(model.predict(input_df)[0])
    conf = float(np.max(model.predict_proba(input_df)[0])) * 100

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"Disease Present (Risk) - Confidence: {conf:.2f}%")
    else:
        st.success(f"No Disease - Confidence: {conf:.2f}%")

    st.subheader("Top 3 Feature Drivers")
    top3 = top_features[:3]
    feat_names = [x["feature"] for x in top3]
    feat_vals = [x["importance"] for x in top3]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(feat_names[::-1], feat_vals[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top 3 Model Drivers")
    st.pyplot(fig)

    explanation = (
        "This prediction is mainly driven by high-impact cardiovascular features such as "
        "ST depression, heart rate response, and chest-pain pattern. "
        "If risk is high, this should be treated as a screening alert and followed by "
        "clinical review and confirmatory testing."
    )
    st.info(explanation)
