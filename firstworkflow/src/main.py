from data_loader import load_images_from_folder
from preprocess import normalize, denoise
from analysis import extract_features, segment_image
from visualization import display_slice

# Path to the folder containing PNG images
data_folder = '../data/Testcases/'  # Update this path if needed

def main():
    # Step 1: Load all PNG images
    images = load_images_from_folder(data_folder)
    if not images:
        print("No images loaded. Please check the folder path and file types.")
        return

    print(f"Loaded {len(images)} PNG images.")

    # Process each image
    for idx, image in enumerate(images):
        print(f"\nProcessing image {idx + 1}/{len(images)}...")

        # Step 2: Normalize the image
        normalized_image = normalize(image)
        print(f"Image {idx + 1} normalized.")

        # Step 3: Denoise the image
        denoised_image = denoise(normalized_image, sigma=1)
        print(f"Image {idx + 1} denoised.")

        # Step 4: Extract features
        features = extract_features(denoised_image)
        print(f"Features extracted for image {idx + 1}: {features}")

        # Step 5: Segment the image
        segmented_image = segment_image(denoised_image, threshold=0.5)
        print(f"Image {idx + 1} segmented.")

        # Step 6: Visualize the results
        display_slice(normalized_image, title=f"Normalized Image {idx + 1}")
        display_slice(denoised_image, title=f"Denoised Image {idx + 1}")
        display_slice(segmented_image, title=f"Segmented Image {idx + 1}")

if __name__ == "__main__":
    main()


