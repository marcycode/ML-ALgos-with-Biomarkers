import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
df_clinical = pd.read_csv("Predicting response to immunotherapy(Sheet5).csv")
df_images = pd.read_csv("image_texture_features.csv")

# Select numeric columns from both datasets for similarity matching
clinical_features = df_clinical[['Age', 'Tumor mutational burden', 'N/L Ratio', 'Platelet count']].fillna(0)
image_features = df_images[['contrast', 'entropy']].fillna(0)

# Normalize data for fair comparison
scaler = StandardScaler()
clinical_scaled = scaler.fit_transform(clinical_features)
image_scaled = scaler.transform(image_features)

# Compute cosine similarity between images and clinical records
similarity_matrix = cosine_similarity(image_scaled, clinical_scaled)

# Assign each image to the most similar clinical record
df_images['assigned_clinical_index'] = similarity_matrix.argmax(axis=1)

# Merge datasets using the assigned index
df_merged = df_images.merge(df_clinical, left_on="assigned_clinical_index", right_index=True, how="left")

# Save merged dataset
df_merged.to_csv("merged_dataset.csv", index=False)
print("Merged dataset saved as 'merged_dataset.csv'")
