
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained model
with open('rf_diabetic_foot_ulcer_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Diabetic Foot Ulcer Risk Prediction")

# User inputs
duration_diabetes = st.number_input("Duration of Diabetes (years)", min_value=0, max_value=50, value=5)

peripheral_neuropathy = st.radio(
    "Peripheral Neuropathy",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

pad = st.radio(
    "Peripheral Arterial Disease (PAD)",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

poor_glycemic_control = st.radio(
    "Poor Glycemic Control",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

foot_deformities = st.radio(
    "Foot Deformities",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

previous_ulcers = st.radio(
    "Previous History of Ulcers",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

pressure = st.slider("Foot Pressure Sensor Value", min_value=0.0, max_value=150.0, value=50.0)

if st.button("Predict Ulcer Risk"):
    input_dict = {
        'duration_diabetes': [duration_diabetes],
        'peripheral_neuropathy': [peripheral_neuropathy],
        'pad': [pad],
        'poor_glycemic_control': [poor_glycemic_control],
        'foot_deformities': [foot_deformities],
        'previous_ulcers': [previous_ulcers],
        'pressure': [pressure]
    }
    input_df = pd.DataFrame(input_dict)
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0, 1]
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    st.markdown(f"### Risk Level: **{risk_label}**")
    st.markdown(f"### Risk Probability: {proba:.2f}")
