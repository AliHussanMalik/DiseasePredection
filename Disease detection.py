import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
import tkinter as tk
from tkinter import messagebox

# Load the dataset
df = pd.read_csv('C:\\Users\\Ali Hussan\\PycharmProjects\\pythonProject7\\The_Cancer_data_1500_V2.csv')

# Visualize the correlation matrix using a heatmap
corr_matrix = df.corr()
plt.figure(figsize=(8, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Prepare data for training
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train different models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Fit and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n" + "-"*60 + "\n")

# Save the best model (Random Forest in this example)
best_model = models["Random Forest"]
joblib.dump(best_model, 'cancer_diagnosis_model.pkl')

# Load the trained model
model = joblib.load('cancer_diagnosis_model.pkl')

# Function to make prediction
def predict():
    try:
        age = int(entry_age.get())
        gender = int(entry_gender.get())
        bmi = float(entry_bmi.get())
        smoking = int(entry_smoking.get())
        genetic_risk = int(entry_genetic_risk.get())
        physical_activity = int(entry_physical_activity.get())
        alcohol_intake = int(entry_alcohol_intake.get())
        cancer_history = int(entry_cancer_history.get())

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'BMI': bmi,
            'Smoking': smoking,
            'GeneticRisk': genetic_risk,
            'PhysicalActivity': physical_activity,
            'AlcoholIntake': alcohol_intake,
            'CancerHistory': cancer_history
        }])

        # Make prediction
        prediction = model.predict(input_data)

        # Show the result in a message box
        diagnosis = 'Positive' if prediction[0] == 1 else 'Negative'
        messagebox.showinfo("Prediction Result", f"Diagnosis: {diagnosis}")
    except Exception as e:
        messagebox.showerror("Input Error", str(e))

# Create the main window
root = tk.Tk()
root.title("Cancer Diagnosis Prediction")

# Create and place labels and entry fields for user input
labels = [
    "Age", "Gender (0=Female, 1=Male)", "BMI (kg/m²)", "Smoking (0=No, 1=Yes)",
    "Genetic Risk (0=Low, 1=High)", "Physical Activity (hours per week)",
    "Alcohol Intake (0=No, 1=Yes)", "Cancer History (0=No, 1=Yes)"
]

entries = {}
for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[label] = entry

# Add placeholder for BMI entry
entries["BMI (kg/m²)"].insert(0, "e.g., 25.0")

entry_age = entries["Age"]
entry_gender = entries["Gender (0=Female, 1=Male)"]
entry_bmi = entries["BMI (kg/m²)"]
entry_smoking = entries["Smoking (0=No, 1=Yes)"]
entry_genetic_risk = entries["Genetic Risk (0=Low, 1=High)"]
entry_physical_activity = entries["Physical Activity (hours per week)"]
entry_alcohol_intake = entries["Alcohol Intake (0=No, 1=Yes)"]
entry_cancer_history = entries["Cancer History (0=No, 1=Yes)"]

# Create and place the Predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

# Start the Tkinter event loop
root.mainloop()
