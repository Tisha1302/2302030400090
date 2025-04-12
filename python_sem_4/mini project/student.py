import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Load data
data = pd.read_csv('student_data.csv')

# Step 2: Convert 'Result' to numeric
data['Result'] = data['Result'].map({'Pass': 1, 'Fail': 0})

# Step 3: Set features and target
X = data[['Study_Hours', 'Attendance', 'Previous_Grades']]
y = data['Result']

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Step 5: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Predict for all students
all_predictions = model.predict(X)

# Step 7: Add predictions to DataFrame
data['Predicted_Result'] = ["Pass" if val == 1 else "Fail" for val in all_predictions]

# Step 8: Show original and predicted results
print(data[['Study_Hours', 'Attendance', 'Previous_Grades', 'Result', 'Predicted_Result']])
