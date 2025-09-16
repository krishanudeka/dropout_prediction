import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("xgboost_dropout_model.pkl")

st.title("🎓 Student Dropout Risk Predictor")

# Input fields
attendance = st.slider("Attendance (%)", 0, 100, 75)
current_gpa = st.number_input("Current GPA", 0.0, 10.0, 7.0)
previous_gpa = st.number_input("Previous GPA", 0.0, 10.0, 7.5)
gpa_diff = current_gpa - previous_gpa
non_prod = st.number_input("Non-Productive Hours per Day", 0, 12, 2)
prod = st.number_input("Productive Hours per Day", 0, 12, 3)
club_score = st.slider("Club Activity Score", 0, 10, 3)
internship = st.selectbox("Internship Status", ["Yes", "No"])
family_income = st.selectbox("Family Income", ["<1 LPA", "1-5 LPA", "5-10 LPA", ">10 LPA"])

internship_val = 1 if internship == "Yes" else 0

# Prepare dataframe
data = {
    'Attendance': attendance,
    'Current_GPA': current_gpa,
    'Previous_GPA': previous_gpa,
    'GPA_Diff': gpa_diff,
    'NonProductive_Hrs': non_prod,
    'Productive_Hrs': prod,
    'Club_Score': club_score,
    'Internship_Status': internship_val,
    'Family_Income': family_income
}
df = pd.DataFrame([data])
df = pd.get_dummies(df, columns=['Family_Income'])
for col in model.feature_names_in_:
    if col not in df.columns:
        df[col] = 0
df = df[model.feature_names_in_]

# Prediction
if st.button("Predict Dropout Risk"):
    prob = model.predict(df)[0]
    prob_percent = round(prob * 100, 1)
    st.metric("Predicted Dropout Probability", f"{prob_percent}%")
