import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')

# File extensions
ALLOWED_EXTENSIONS = ['dcm', 'nii', 'nii.gz']

# Example constant
THRESHOLD_VALUE = 0.5
