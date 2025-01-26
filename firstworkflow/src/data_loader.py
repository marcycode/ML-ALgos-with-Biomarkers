import os
from PIL import Image
import numpy as np

def load_png(filepath):
    """Load a PNG file as a NumPy array."""
    try:
        # Open the PNG image using Pillow
        image = Image.open(filepath)
        # Convert to a NumPy array for processing
        image_array = np.array(image)
        return image_array
    except Exception as e:
        print(f"Error loading PNG file: {e}")
        return None

def load_images_from_folder(folder_path):
    """Load all PNG images from a given folder."""
    images = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                file_path = os.path.join(folder_path, filename)
                image = load_png(file_path)
                if image is not None:
                    images.append(image)
        return images
    except Exception as e:
        print(f"Error loading images from folder: {e}")
        return []

# Example usage
if __name__ == "__main__":
    folder = '../data/Testcases/'  # Path to folder with PNG files
    images = load_images_from_folder(folder)
    print(f"Loaded {len(images)} images")
