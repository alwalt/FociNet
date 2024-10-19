import os
import pandas as pd

# Parameters
image_dir = 'BPS-Microscopy/train/'  # Path to your image directory
metadata_path = 'BPS-Microscopy/train/meta.csv'  # Path to metadata CSV
output_file = 'metadata_validation_log.txt'  # File to log metadata validation results

# Load metadata
metadata = pd.read_csv(metadata_path)

print(metadata[:10])

# Open the log file
with open(output_file, 'w') as log_file:
    
    # Check for missing or null values
    log_file.write("Checking for missing or null values...\n")
    missing_values = metadata.isnull().sum()
    if missing_values.any():
        log_file.write(f"\nColumns with missing values:\n{missing_values}\n")
    else:
        log_file.write("No missing values found.\n")
    
    # Check if particle types are valid
    log_file.write("\nChecking for invalid particle types...\n")
    valid_particle_types = {'Fe', 'X-ray', 'Control'}
    invalid_particle_types = metadata[~metadata['particle_type'].isin(valid_particle_types)]
    if not invalid_particle_types.empty:
        log_file.write(f"Invalid particle types found:\n{invalid_particle_types[['filename', 'particle_type']]}\n")
    else:
        log_file.write("All particle types are valid.\n")
    
    # Check if dose values are numeric and within a reasonable range
    log_file.write("\nChecking for invalid dose values...\n")
    metadata['dose_Gy'] = pd.to_numeric(metadata['dose_Gy'], errors='coerce')  # Convert to numeric, forcing errors to NaN
    invalid_doses = metadata[(metadata['dose_Gy'].isna()) | (metadata['dose_Gy'] < 0)]
    if not invalid_doses.empty:
        log_file.write(f"Invalid dose values found:\n{invalid_doses[['filename', 'dose_Gy']]}\n")
    else:
        log_file.write("All dose values are valid and non-negative.\n")
    
    # Check if file paths exist
    log_file.write("\nChecking if all file paths exist...\n")
    missing_files = []
    for filepath in metadata['filename']:
        full_path = os.path.join(image_dir, filepath)
        if not os.path.exists(full_path):
            missing_files.append(full_path)
    if missing_files:
        log_file.write(f"Missing files:\n{missing_files}\n")
    else:
        log_file.write("All files exist.\n")

print(f"Metadata validation results logged to {output_file}")
