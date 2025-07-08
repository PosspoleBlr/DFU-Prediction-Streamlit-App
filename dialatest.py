import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pickle

# --------------------------
# Step 1: Generate synthetic dataset
# --------------------------
np.random.seed(42)
n_samples = 1000

duration_diabetes = np.random.randint(0, 31, size=n_samples)
peripheral_neuropathy = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
pad = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
poor_glycemic_control = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
foot_deformities = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
previous_ulcers = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
pressure = np.random.normal(loc=50, scale=15, size=n_samples)

risk_prob = (
    0.02 * duration_diabetes +
    0.3 * peripheral_neuropathy +
    0.25 * pad +
    0.2 * poor_glycemic_control +
    0.15 * foot_deformities +
    0.4 * previous_ulcers +
    0.01 * (pressure - 40)
)
risk_prob = np.clip(risk_prob, 0, 1)
ulcer_risk = np.random.binomial(1, risk_prob)

data = pd.DataFrame({
    'duration_diabetes': duration_diabetes,
    'peripheral_neuropathy': peripheral_neuropathy,
    'pad': pad,
    'poor_glycemic_control': poor_glycemic_control,
    'foot_deformities': foot_deformities,
    'previous_ulcers': previous_ulcers,
    'pressure': pressure,
    'ulcer_risk': ulcer_risk
})

csv_filename = 'synthetic_diabetic_foot_ulcer_data_new.csv'
data.to_csv(csv_filename, index=False)
print(f"Synthetic dataset saved to '{csv_filename}'")

# --------------------------
# Step 2: Load dataset and split
# --------------------------
loaded_data = pd.read_csv(csv_filename)
X = loaded_data.drop(columns=['ulcer_risk'])
y = loaded_data['ulcer_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Step 3: Train model
# --------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --------------------------
# Step 4: Evaluate model in terminal only
# --------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print("Classification Report:\n", report)

# --------------------------
# Step 5: Save model
# --------------------------
with open('rf_diabetic_foot_ulcer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# --------------------------
# Step 6: Generate Streamlit app code (no metrics shown in app)
# --------------------------
streamlit_app_code = '''
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
'''

# Save the Streamlit code to app.py
with open('app.py', 'w') as f:
    f.write(streamlit_app_code)

print("\n✅ Streamlit app code saved to 'app.py'.")
print("👉 Run the app locally with: streamlit run app.py")
