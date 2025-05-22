import numpy as np
import os

# Define the amount to shift z-values (this value is in cm).
z_shift = 2
z_shift_str = str(z_shift).replace('.', '-')

# Input and output directories.
input_directory = '/home/nstrobbe/seren036/hgcalml/hgcal_minimal_eval_example/cleanevents/singleTau24-11-25_E50/'
output_directory = f'/home/nstrobbe/seren036/hgcalml/hgcal_minimal_eval_example/modifiedevents/25-05-21/singleTau24-11-25_E50_zshifted2cm/'

# Ensure the output directory exists and make one if there isn't one already.
os.makedirs(output_directory, exist_ok=True)

# Process each .npz file in the input directory.
for file_name in os.listdir(input_directory):
    if file_name.endswith('.npz'):
        file_path = os.path.join(input_directory, file_name)
        data = np.load(file_path, allow_pickle=True)
        modified_data = {}

        # Handle rechits_keys.
        rechits_keys = data['rechits_keys']
        z_key_index = np.where(rechits_keys == 'RecHitHGC_z')[0][0]

        rechits_array = data['rechits_array']
        z_data = rechits_array[:, z_key_index]

        # Apply the shift to the z-values.
        rechits_array[:, z_key_index] += z_shift

        # Update the modified data dictionary.
        modified_data['rechits_keys'] = rechits_keys
        modified_data['rechits_array'] = rechits_array

        # Copy any other keys in the dataset as is.
        for key in data.files:
            if key not in ['rechits_keys', 'rechits_array']:
                modified_data[key] = data[key]

        # Define the output file name
        output_file_name = file_name.replace('.npz', f'_zshifted{z_shift_str}cm.npz')
        output_file_path = os.path.join(output_directory, output_file_name)
        
        # Save the modified data to a new .npz file and print confirmation.
        np.savez(output_file_path, **modified_data)

        print(f"Modified file saved as: {output_file_path}")
