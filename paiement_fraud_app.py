import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

model = joblib.load("data/fraud_detection_model/model.jb")
encoders = joblib.load("data/fraud_detection_model/label_encoders.jb")
scalers = joblib.load("data/fraud_detection_model/scalers_encoder.jb")

def haversine(lat1, long1, lat2, long2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(long2 - long1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return (R * c) / 1000.0   # return km

st.title("Fraud Detection System")
st.write("Enter the Transaction details below")

col1, col2, col3 = st.columns(3)

with col1:
    merchant   = st.text_input("Merchant", "fraud_Rippin, Kub and Mann")
    category   = st.text_input("Category", "misc_net")
    gender     = st.selectbox("Gender", ["M", "F"], index=1)
    city       = st.text_input("City", "Moravian Falls")
    state      = st.text_input("State", "NC")

with col2:
    job        = st.text_input("Job Title", "Psychologist, counselling")
    cc_num     = st.text_input("Credit Card Number", "2703186189652095")
    amt        = st.number_input("Amount ($)", min_value=0.0, value=4.97, format="%.2f")
    city_pop   = st.number_input("City Population", 0, value=3495)
    birth_year = st.number_input("Birth Year", 1900, 2025, value=1988)

with col3:
    date       = st.date_input("Transaction Date", datetime(2012, 1, 1))
    lat        = st.number_input("User Lat", format="%.6f", value=36.4583)
    long       = st.number_input("User Long", format="%.6f", value=-80.9841)
    merch_lat  = st.number_input("Merchant Lat", format="%.6f", value=36.0)
    merch_long = st.number_input("Merchant Long", format="%.6f", value=-80.0)

# Derived features
distance = haversine(lat, long, merch_lat, merch_long)
hour = 0 if date is None else date.hour if hasattr(date, "hour") else 0
day = date.day
month = date.month
unix_time = int(datetime.combine(date, datetime.min.time()).timestamp())

# Build input DataFrame
input_data = pd.DataFrame([{
    'merchant': merchant,
    'category': category,
    'gender': gender,
    'cc_num': int(cc_num),
    'city': city,
    'state': state,
    'job': job,
    'amount': int(amt),
    'city_pop': int(city_pop),
    'unix_time': unix_time,
    'trans_hour': hour,
    'trans_day': day,
    'trans_month': month,
    'birth_year': birth_year,
    'distance': distance
}])
# Encode categorical features
categorical_features = ['merchant', 'category', 'gender', 'city', 'state', 'job']
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
continuous_features = ['amount', 'city_pop', 'unix_time', 'trans_hour', 'trans_day', 'trans_month', 'birth_year', 'distance']
for col in continuous_features:
    if col in scalers:
        input_data[col] = scalers[col].transform(input_data[[col]])
    else:
        input_data[col] = input_data[col]

# Predict
if st.button("Check for Fraud"):
    prediction = model.predict(input_data)[0]
    print( prediction)
    result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
    st.subheader(f"Prediction: {result}")

