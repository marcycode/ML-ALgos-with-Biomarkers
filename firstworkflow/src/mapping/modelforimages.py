from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
import numpy as np
from skimage.color import rgb2gray
import os
import pandas as pd

# Function to extract texture features
def extract_texture_features(img_path):
    img = imread(img_path, as_gray=True)
    glcm = graycomatrix(img.astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = np.sum(img * np.log2(img + 1e-9))  # Entropy calculation
    return [contrast, entropy]

# Process all images
image_folder = "../../data/Testcases/"
image_data = []
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        features = extract_texture_features(img_path)
        image_data.append([filename] + features)

# Convert to DataFrame
df_features = pd.DataFrame(image_data, columns=["image_filename", "contrast", "entropy"])

# Save to CSV
df_features.to_csv("image_texture_features.csv", index=False)

print(f"Extracted texture features for {len(df_features)} images.")
