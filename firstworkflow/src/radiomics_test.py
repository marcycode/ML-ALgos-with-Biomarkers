import os

import cv2

import numpy as np
import pandas as pd
import skimage.io as io
import skimage.feature as feature
import skimage.filters as filters
import skimage.measure as measure
from tqdm import tqdm

# 🔹 Folder containing lung scan images
image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Testcases"))


# 🔹 List to store extracted features
data = []

# 🔹 Process each image
for image_file in tqdm(os.listdir(image_dir), desc="Processing lung scans"):
    if image_file.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_dir, image_file)
        
        # Load image in grayscale
        image = io.imread(image_path, as_gray=True)
        
        # Apply Gaussian filter to remove noise
        image_smoothed = filters.gaussian(image, sigma=1)

        # Compute Intensity Features
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)

        # Compute Texture Features (Haralick)
        glcm = feature.graycomatrix(image.astype(np.uint8), [1], [0], symmetric=True, normed=True)
        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
        energy = feature.graycoprops(glcm, 'energy')[0, 0]
        homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]

        # Compute Shape Features (Find Contours)
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = max([cv2.contourArea(c) for c in contours], default=0)

        # Store features
        data.append({
            "Image_Name": image_file,
            "Mean_Intensity": mean_intensity,
            "Std_Intensity": std_intensity,
            "GLCM_Contrast": contrast,
            "GLCM_Energy": energy,
            "GLCM_Homogeneity": homogeneity,
            "Largest_Contour_Area": largest_area
        })

# 🔹 Convert to DataFrame
df = pd.DataFrame(data)

# 🔹 Save to CSV
df.to_csv("lung_scan_features.csv", index=False)
print("\n✅ Features saved to 'lung_scan_features.csv'!")
