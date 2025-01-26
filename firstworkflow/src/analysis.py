def extract_features(image):
    """Extract simple statistical features."""
    return {
        'mean': image.mean(),
        'std_dev': image.std(),
        'min': image.min(),
        'max': image.max(),
    }

def segment_image(image, threshold=0.5):
    """Segment the image using a simple threshold."""
    return (image > threshold).astype(int)
