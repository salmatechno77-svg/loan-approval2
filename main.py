import joblib
import pandas as pd

# Loading the model
model = joblib.load("loan_approval_v.1.pkl")

# Read the input data from the CSV file
df = pd.read_csv("input_data.csv")

# Make predictions using the loaded model
y_pred = model.predict(df)

labels = ["approved" if x == 1 else "rejected" for x in y_pred]

# Print the prediction results
print("Predictions:")
print(labels)