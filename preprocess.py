import os
import cv2
import pandas as pd

# Parameters
image_dir = 'BPS-Microscopy/train/'  # Path to your image directory
metadata_path = 'BPS-Microscopy/train/meta.csv'  # Metadata CSV
output_file = 'image_info_log.txt'  # File to log image information

# Load metadata
metadata = pd.read_csv(metadata_path)

# Filter for valid TIFF images
metadata = metadata[metadata['filename'].str.endswith('.tif')]

# Add full path to filenames
metadata['filepath'] = metadata['filename'].apply(lambda x: os.path.join(image_dir, x))

# Function to load and inspect image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if the image is None or empty
    if image is None or image.size == 0:
        return None, None
    
    # Detect if image has more than 2 channels (should only be (x, y) for grayscale)
    if image.ndim == 3:
        return None, 'multi_channel'

    # Check if image is completely black (all pixels are zero)
    if not image.any():  # `not image.any()` returns True if all pixel values are zero
        return None, 'black_image'

    # Return image and its shape
    return image, image.shape

# Open the log file
with open(output_file, 'w') as log_file:
    log_file.write("Image Path, Status, Shape\n")
    
    # Iterate over all images and log the results
    for index, row in metadata.iterrows():
        image_path = row['filepath']
        
        # Load image and get its shape
        image, shape = load_image(image_path)
        
        if image is None:
            if shape == 'multi_channel':
                log_file.write(f"{image_path}, Multi-channel, N/A\n")
            elif shape == 'black_image':
                log_file.write(f"{image_path}, Black Image, N/A\n")
            else:
                log_file.write(f"{image_path}, Corrupt, N/A\n")
        else:
            log_file.write(f"{image_path}, Valid, {shape}\n")

print(f"Image information has been logged to {output_file}")
