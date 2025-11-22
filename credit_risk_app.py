import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("data/credit_risk_model/model.jb")
encoders = joblib.load("data/credit_risk_model/label_encoders.jb")
scalers = joblib.load("data/credit_risk_model/scalers_encoder.jb")

st.title("Credit Risk Prediction System")
st.write("Enter the Transaction details Below")

col1, col2, col3 = st.columns(3)

with col1:
    age                 = st.number_input("Age", min_value=0, max_value=150, value=35)
    income              = st.number_input("Annual Income ($)", min_value=0.0, value=55000.0, format="%.2f")
    home_ownership      = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], index=0)

with col2:
    employment_length   = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
    purpose             = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], index=0)
    amount              = st.number_input("Loan Amount ($)", min_value=0.0, value=15000.0, format="%.2f")

with col3:
    interest_rate       = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.5, format="%.2f")
    status              = st.selectbox("Loan Status", ["0", "1"], index=0)
    percent_income      = st.number_input("Percent of Income (%)", min_value=0.0, max_value=100.0, value=15.0, format="%.2f")


# Build input DataFrame
input_data = pd.DataFrame([{
    'age': int(age),
    'income': float(income),
    'home_ownership': home_ownership,
    'employment_length': int(employment_length),
    'purpose': purpose,
    'amount': float(amount),
    'interest_rate': float(interest_rate),
    'status': status,
    'percent_income': float(percent_income)
}])
# Encode categorical features
categorical_features = ['home_ownership', 'purpose', 'status']
for col in categorical_features:
    if col in encoders:
        try:
            input_data[col] = encoders[col].transform(input_data[col])
        except ValueError:
            # handle unseen categories
            input_data[col] = -1
    else:
        input_data[col] = -1
# Scale numerical features
continuous_features = ['age', 'income', 'employment_length', 'amount', 'interest_rate', 'percent_income']
for col in continuous_features:
    if col in scalers:
        input_data[col] = scalers[col].transform(input_data[[col]])
    else:
        input_data[col] = input_data[col]

# Predict
if st.button("Check For Credit Risk"):
    prediction = model.predict(input_data)[0]
    result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
    st.subheader(f"Prediction: {result}")

