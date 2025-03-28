import os
import pandas as pd

# Define dataset path
image_folder = "../../data/Testcases/"



# Extract file names and corresponding labels
image_data = []
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        # Extract label based on file structure (modify based on actual folder structure)
        if "normal" in filename.lower():
            label = "Healthy"
        elif "benign" in filename.lower():
            label = "Benign"
        elif "malignant" in filename.lower():
            label = "Malignant"
        else:
            continue

        image_data.append([filename, label])

# Convert to DataFrame
df_images = pd.DataFrame(image_data, columns=["image_filename", "diagnosis"])

# Save mapping file for reference
df_images.to_csv("image_metadata.csv", index=False)
