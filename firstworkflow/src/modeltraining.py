import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
file_path = "merged_patient_data.csv"  # Ensure correct path
df = pd.read_csv(file_path)
print("Unique values in Stage_x:", df["Stage_x"].unique())
print("Unique values in Stage_y:", df["Stage_y"].unique())

df["Stage_Final"] = df["Stage_y"].fillna(df["Stage_x"])

# Convert Stage into binary labels
df["Has_Lung_Cancer"] = df["Stage_Final"].apply(lambda x: 1 if str(x) in ["IIIA", "IIIB", "IV"] else 0)

# Drop unnecessary columns
df.drop(columns=["Stage_x", "Stage_y", "Stage_Final"], inplace=True)


# Drop non-useful columns (like Patient_ID if exists)
df.drop(columns=["Patient_ID"], inplace=True, errors="ignore")

# Separate numerical and categorical columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Convert numerical columns properly (to handle strings in numeric fields)
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numbers, replace invalids with NaN

# Fill missing values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())  # Fill numeric NaN with mean
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])  # Fill categorical NaN with mode

# Encode categorical variables
df = pd.get_dummies(df, columns=categorical_cols)

# Define Target & Features
target_column = "Has_Lung_Cancer"  # Change if different
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target variable

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training & test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(df["Has_Lung_Cancer"].value_counts(normalize=True))


# Save Model
joblib.dump(model, "lung_cancer_detector.pkl")
print("\n💾 Model saved as 'lung_cancer_detector.pkl'")
