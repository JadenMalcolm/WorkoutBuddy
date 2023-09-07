import os
import numpy as np
import DataPoints as keypoints_data

def parse_key_points(numpy_folder, keypoints_data):
    # Convert the keypoints data to a structured numpy array
    dt = np.dtype([('all_points_x', 'O'), ('all_points_y', 'O')])  # Use 'O' for flexible type
    keypoints_np = np.array([(data['all_points_x'], data['all_points_y']) for data in keypoints_data], dtype=dt)

    # Save the keypoints numpy array to a file
    keypoints_path = os.path.join(numpy_folder, "keypoints_data.npy")
    np.save(keypoints_path, keypoints_np)

# Sample keypoints data (list of dictionaries)

# Specify the folder where you want to save the numpy array
numpy_folder = "/path/to/your/folder"

# Call the function to parse and save keypoints
parse_key_points(numpy_folder, keypoints_data)




