import numpy as np
import os

# Input and output directories
input_directory = '/home/nstrobbe/seren036/hgcalml/hgcal_minimal_eval_example/cleanevents/singleTau24-11-25_E50'
output_directory = '/home/nstrobbe/seren036/hgcalml/hgcal_minimal_eval_example/modifiedevents/25-05-04/singleTau24-11-25_E50_Layer7-11Removed/' 

# Ensure the output directory exists and makes one if there isn't one already. This is just to save me time because creating and then matching the really long names gets tiring.
os.makedirs(output_directory, exist_ok=True)

# Layers to remove (list or range works)
layers_to_remove = [7,8,9,10,11]  

# This is a list of the 'exact' z-position for the layers
layer_positions = np.array([
    322, 323, 325, 326, 328, 329, 331, 332, 
    334, 335, 337, 338, 340, 341, 343, 344, 
    346, 347, 349, 350, 352, 353, 355, 356, 
    358, 359, 361, 362, 368, 373, 379, 384, 
    389, 395, 400, 406, 411, 417, 422, 428, 
    436, 445, 453, 462, 470, 479, 487, 496, 
    505, 513
])

# Get the z-positions of the layers to remove
z_to_remove = layer_positions[np.array(layers_to_remove) - 1]

# Process each .npz file in the input directory
for file_name in os.listdir(input_directory):
    if file_name.endswith('.npz'):
        file_path = os.path.join(input_directory, file_name)
        data = np.load(file_path, allow_pickle=True)
        filtered_data = {}

        # Handle rechits_keys
        rechits_keys = data['rechits_keys']  # Load the array of column names
        z_key_index = np.where(rechits_keys == 'RecHitHGC_z')[0][0]  
        # print(f"Processing file: {file_name}") 
        # print(f"Index of 'RecHitHGC_z': {z_key_index}") # Print statement just because I wanted to check this matched our notation.

        rechits_array = data['rechits_array']
        z_data = rechits_array[:, z_key_index]
        rounded_z = np.round(z_data).astype(int)

        # Create a mask to exclude hits associated with the specified layers
        mask = ~np.isin(rounded_z, z_to_remove)
        print(f"File: {file_name}, Total hits to remove: {np.sum(~mask)}")

        filtered_rechits_array = rechits_array[mask]

        # Update the filtered data dictionary
        filtered_data['rechits_keys'] = rechits_keys
        filtered_data['rechits_array'] = filtered_rechits_array

        # Copy any other keys in the dataset as is. There may be an easier way to do this but I implemented this as a fix of the code I already had.
        for key in data.files:
            if key not in ['rechits_keys', 'rechits_array']:
                filtered_data[key] = data[key]

        # Define the output file name. This is not necessary but it makes it easier to tell if the right files are in the right folders when we're looking at things in retrospect.
        output_file_name = file_name.replace('.npz', '_Layer7-11Removed.npz')
        output_file_path = os.path.join(output_directory, output_file_name)

        # Save the modified data to a new .npz file and print confirmation.
        np.savez(output_file_path, **filtered_data)

        print(f"Modified file saved as: {output_file_path}")