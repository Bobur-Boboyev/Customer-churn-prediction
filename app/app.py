from sklearn.preprocessing import LabelEncoder
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')



@st.cache_resource
def load_model(relative_path):
    base_dir = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(base_dir, relative_path))
    return joblib.load(model_path)

model = load_model("../models/lr_model.pkl")
scaler = load_model("../models/scaler.pkl")
le_dict = load_model("../models/LabelEncoder.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="👜")
st.title("📉 Customer Churn Prediction App")

st.markdown("""
This app predicts whether a customer is likely to **churn** (leave your service) based on their information.
""")
st.sidebar.title("Sidebar")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen?", ["No", "Yes"])
Partner = st.sidebar.selectbox("Has Partner?", ["No", "Yes"])
Dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
PhoneService = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check", 
    "Mailed check", 
    "Bank transfer (automatic)", 
    "Credit card (automatic)"
])
MonthlyCharges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
TotalCharges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
data = {
        "gender": gender,
        "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

def main():
    df = pd.DataFrame([data])

    for col in df.select_dtypes(include='object').columns:
        if col in le_dict:
            df[col] = le_dict[col].transform(df[col])
    st.write("Encoded input to model:")


    scaled_input = scaler.transform(df)
    return scaled_input

if st.sidebar.button("🔮 Predict"):
    try:
        input_data = main()
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("❌ This customer is likely to churn.")
        else:
            st.success("✅ This customer is likely to stay.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

