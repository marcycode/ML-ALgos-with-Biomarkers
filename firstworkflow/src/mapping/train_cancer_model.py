import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


# Load dataset
file_path = "Predicting_response_to_immunotherapy.csv"  # Update with correct filename if needed
df = pd.read_csv(file_path, dtype=str)  # Read everything as string first to clean up

print(df["Has_Lung_Cancer"].value_counts())


# Remove unwanted characters and whitespaces
df = df.map(lambda x: x.strip().replace("\\", "") if isinstance(x, str) else x)

# Convert numeric columns properly
numeric_cols = ["Age", "Tumor mutational burden", "Absolute Neutrophil count",
                "Absolute Lymphocyte count", "N/L Ratio", "Platelet count", "Performance status ECOG"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float, replace errors with NaN

# Drop rows with missing values after conversion
df.dropna(inplace=True)

# Encode categorical variables using one-hot encoding
categorical_cols = ["Gender", "Histology", "PDL1 status", "Genetic mutations", "1st line treatment"]
df = pd.get_dummies(df, columns=categorical_cols)

# Target Variable: Assume **Stage III/IV = Higher Lung Cancer Risk**
df["Has_Lung_Cancer"] = df["Stage"].apply(lambda x: 1 if "III" in str(x) or "IV" in str(x) else 0)

# Drop original "Stage" column
df.drop(columns=["Stage"], inplace=True)

# Normalize numerical features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Check for class imbalance
print("\nClass Distribution:")
print(df["Has_Lung_Cancer"].value_counts(normalize=True))

# Compute and visualize correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Define Features (X) and Target (y)
X = df.drop(columns=["Has_Lung_Cancer"])  # All features except the target
y = df["Has_Lung_Cancer"]  # Target label (1 = Cancer, 0 = No Cancer)

# Split data into training (80%) and testing (20%) sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"\nCross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
print("\nFeature Importances:")
print(feature_importances)

# Save the trained model
joblib.dump(model, "lung_cancer_model.pkl")

# Save feature importances
feature_importances.to_csv("feature_importances.csv", index=False)

print("\n✅ Model training complete! Model saved as 'lung_cancer_model.pkl'.")