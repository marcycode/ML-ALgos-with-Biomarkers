import numpy as np
from scipy.ndimage import gaussian_filter

def normalize(image):
    """Normalize an image to the range [0, 1]."""
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def denoise(image, sigma=1):
    """Apply Gaussian smoothing for noise reduction."""
    return gaussian_filter(image, sigma=sigma)
