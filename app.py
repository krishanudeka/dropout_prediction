import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("xgboost_dropout_model.pkl")

# Suggestion function
def get_suggestions(student, predicted_prob):
    suggestions = []
    if student["Attendance"] < 60:
        suggestions.append("⚠️ Low attendance, risk of debarment. Improve daily routine.")
    if student["Current_GPA"] < 6:
        suggestions.append("📚 GPA is low. Try remedial classes or study groups.")
    if student["GPA_Diff"] < -0.5:
        suggestions.append("📉 GPA dropped. Explore new study methods or counseling.")
    if student["NonProductive_Hrs"] > 4:
        suggestions.append("🧠 Too many unproductive hours. Reduce screen time, balance activities.")
    if student["Productive_Hrs"] < 1:
        suggestions.append("⏰ Low study time. Use time management techniques.")
    if student["Club_Score"] < 2:
        suggestions.append("🎭 Limited extracurriculars. Join clubs for social and skill growth.")
    if student["Internship_Status"] == 0:
        suggestions.append("💼 No internship. Apply for internships to gain experience.")
    if student["Family_Income"] in ['<1 LPA', '1-5 LPA']:
        suggestions.append("💰 Financial risk. Look for scholarships or aid.")
    if predicted_prob > 0.8:
        suggestions.append("🚨 High dropout risk! Meet advisor for support.")
    if not suggestions:
        suggestions.append("✅ Student is performing well. Keep up the good work.")
    return suggestions

# Streamlit UI
st.title("🎓 Student Dropout Prediction App")

st.subheader("Enter Student Details")

attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
current_gpa = st.number_input("Current GPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
previous_gpa = st.number_input("Previous GPA", min_value=0.0, max_value=10.0, value=7.6, step=0.1)
gpa_diff = current_gpa - previous_gpa
non_productive = st.number_input("Non-Productive Hours (per day)", min_value=0.0, max_value=24.0, value=2.0, step=0.1)
semi_productive = st.number_input("Semi-Productive Hours (per day)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
productive = st.number_input("Productive Hours (per day)", min_value=0.0, max_value=24.0, value=4.0, step=0.1)
club_score = st.number_input("Club/Extracurricular Score", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
internship_status = st.selectbox("Internship Completed?", [0, 1])
family_income = st.selectbox("Family Income", ['<1 LPA', '1-5 LPA', '5-10 LPA', '10+ LPA'])

if st.button("🔮 Predict Dropout Probability"):
    student = {
        "Attendance": attendance,
        "Current_GPA": current_gpa,
        "Previous_GPA": previous_gpa,
        "GPA_Diff": gpa_diff,
        "NonProductive_Hrs": non_productive,
        "SemiProductive_Hrs": semi_productive,
        "Productive_Hrs": productive,
        "Club_Score": club_score,
        "Internship_Status": internship_status,
        "Family_Income": family_income
    }

    # Prepare data for model
    student_df = pd.DataFrame([student])
    student_df = pd.get_dummies(student_df, columns=["Family_Income"])
    for col in model.feature_names_in_:
        if col not in student_df.columns:
            student_df[col] = 0
    student_df = student_df[model.feature_names_in_]

    # Predict
    pred_prob = model.predict(student_df)[0]
    pred_prob = np.clip(pred_prob, 0, 1)

    # Show results
    st.metric("📊 Dropout Probability", f"{pred_prob * 100:.1f}%")
    st.subheader("🎯 Personalized Suggestions")
    for s in get_suggestions(student, pred_prob):
        st.write(f"- {s}")
