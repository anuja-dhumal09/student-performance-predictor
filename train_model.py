import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("dataset.csv")

# Convert Result into numbers
encoder = LabelEncoder()
data["Result"] = encoder.fit_transform(data["Result"])

# Input and output
X = data[["Study_Hours", "Attendance", "Previous_Marks", "Assignment_Score"]]
y = data["Result"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Take student details
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance (%): "))
previous_marks = float(input("Enter previous marks: "))
assignment_score = float(input("Enter assignment score: "))

# Make prediction
prediction = model.predict([
    [study_hours, attendance, previous_marks, assignment_score]
])

# Show result
result = encoder.inverse_transform(prediction)

print("Predicted Result:", result[0])