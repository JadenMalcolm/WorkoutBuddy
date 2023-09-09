from DataPoints import original_point_data, image_resolutions_test

scaled_data = []  # To store the scaled data
current_resolution_idx = 0  # Index to keep track of the current resolution
data_count = 0  # Count of data points processed for the current resolution

# Number of data points for each resolution
data_points_per_resolution = 19

# Iterate through the data
for data in original_point_data:
    # Get the resolution based on the current index
    resolution = image_resolutions_test[current_resolution_idx]
    print(f"Current Resolution Index: {current_resolution_idx}")
    print(f"Current Resolution: {resolution}")

    if "cx" in data and "cy" in data:
        original_width, original_height = resolution
        scale_x = 224 / original_width
        scale_y = 224 / original_height
        data["cx"] = int(data["cx"] * scale_x)
        data["cy"] = int(data["cy"] * scale_y)

    scaled_data.append(data)
    data_count += 1

    if data_count >= data_points_per_resolution:
        # If the count exceeds the expected number of data points for this resolution, increment the resolution index
        current_resolution_idx += 1
        data_count = 0  # Reset the data count for the new resolution

# Print the scaled data
for data in scaled_data:
    print(f"{data},")
