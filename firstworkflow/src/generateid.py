import pandas as pd
import numpy as np
import uuid

# Load the clinical dataset

clinical_file = "mapping/Predicting_response_to_immunotherapy.csv"
clinical_df = pd.read_csv(clinical_file)

# Load the lung scan features dataset
scan_file = "lung_scan_features.csv"
scan_df = pd.read_csv(scan_file)

# Generate unique patient IDs
num_patients = max(len(clinical_df), len(scan_df))
patient_ids = [str(uuid.uuid4())[:8] for _ in range(num_patients)]  # Shorten UUID for readability

# Assign IDs to both datasets
clinical_df["Patient_ID"] = patient_ids[:len(clinical_df)]
scan_df["Patient_ID"] = patient_ids[:len(scan_df)]

# Generate synthetic patient characteristics for scan dataset
np.random.seed(42)
synthetic_data = {
    "Age": np.random.randint(40, 80, size=len(scan_df)),
    "Gender": np.random.choice(["M", "F"], size=len(scan_df)),
    "Histology": np.random.choice(["Adenocarcinoma", "Squamous Cell Carcinoma", "Large Cell Carcinoma"], size=len(scan_df)),
    "Stage": np.random.choice(["IIIA", "IIIB", "IV"], size=len(scan_df)),
    "PDL1 status": np.random.choice(["Positive", "Negative"], size=len(scan_df)),
    "Tumor mutational burden": np.random.randint(1, 10, size=len(scan_df)),
    "Genetic mutations": np.random.choice(["EGFR", "KRAS", "TP53", "BRAF", "MET", "ALK", "ROS1", "RET", "STK11"], size=len(scan_df)),
    "Performance status ECOG": np.random.randint(0, 4, size=len(scan_df)),
}

# Add synthetic patient characteristics to scan dataset
for key, values in synthetic_data.items():
    scan_df[key] = values

# Merge both datasets
merged_df = pd.merge(scan_df, clinical_df, on="Patient_ID", how="outer")

# Save the updated datasets
clinical_df.to_csv("updated_clinical_data.csv", index=False)
scan_df.to_csv("updated_scan_data.csv", index=False)
merged_df.to_csv("merged_patient_data.csv", index=False)

print("✅ Synthetic patient IDs and data added successfully!")
