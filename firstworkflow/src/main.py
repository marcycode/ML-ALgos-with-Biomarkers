import os
import pandas as pd
from data_loader import load_images_from_folder
from preprocess import normalize, denoise
from analysis import extract_features, segment_image

# Path to the folder containing PNG images
data_folder = '../data/Testcases/'  # Update this path if needed
output_csv = "image_std_dev.csv"  # CSV file to store std_dev data

def main():
    # Step 1: Load all PNG images
    images = load_images_from_folder(data_folder)
    if not images:
        print("No images loaded. Please check the folder path and file types.")
        return

    print(f"Loaded {len(images)} PNG images.")

    # Store results
    image_data = []

    # Process each image automatically (no user interaction needed)
    for idx, image in enumerate(images):
        print(f"\nProcessing image {idx + 1}/{len(images)}...")

        # Step 2: Normalize the image
        normalized_image = normalize(image)

        # Step 3: Denoise the image
        denoised_image = denoise(normalized_image, sigma=1)

        # Step 4: Extract features (including std_dev)
        features = extract_features(denoised_image)

        # Extract std_dev from features dictionary
        std_dev = features.get("std_dev", None)

        # Store result
        image_data.append([f"image_{idx + 1}.png", std_dev])  # Save image filename + std_dev

        print(f"Processed image {idx + 1}: std_dev = {std_dev}")

    # Save extracted std_dev values to CSV
    df = pd.DataFrame(image_data, columns=["image_filename", "std_dev"])
    df.to_csv(output_csv, index=False)

    print(f"\n✅ All images processed. Saved std_dev values to {output_csv}")

if __name__ == "__main__":
    main()
