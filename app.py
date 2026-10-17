import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("dataset.csv")

# Convert Result into numbers
encoder = LabelEncoder()
data["Result"] = encoder.fit_transform(data["Result"])

# Prepare data
X = data[["Study_Hours", "Attendance", "Previous_Marks", "Assignment_Score"]]
y = data["Result"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# App title
st.title("🎓 Student Performance Predictor")
st.subheader("Predict student performance using Machine Learning")

st.write("Enter student details below and click Predict Result.")

# User inputs
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=15.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
previous_marks = st.number_input("Previous Marks", min_value=0.0, max_value=100.0)
assignment_score = st.number_input("Assignment Score", min_value=0.0, max_value=100.0)

# Prediction
if st.button("Predict Result"):
    prediction = model.predict([
        [study_hours, attendance, previous_marks, assignment_score]
    ])

    result = encoder.inverse_transform(prediction)

    st.success(f"Predicted Result: {result[0]}")
    st.info("This project uses Machine Learning to predict student performance based on study hours, attendance, previous marks and assignment score.")