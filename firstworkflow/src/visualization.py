import matplotlib.pyplot as plt

def display_slice(image, title="Image Slice"):
    """Display a 2D slice of the image."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()
