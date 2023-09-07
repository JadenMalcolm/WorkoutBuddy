from DataPoints import original_point_data, image_resolutions_test

scaled_data = []  # To store the scaled data
current_resolution_idx = 0  # Index to keep track of the current resolution

# Iterate through the data
for data in original_point_data:
    if "correct" in data:
        # If "correct" is found, increment the resolution index
        current_resolution_idx += 1
        if current_resolution_idx >= len(image_resolutions_test):
            break  # Stop if there are no more resolutions

    # Get the resolution based on the current index
    resolution = image_resolutions_test[current_resolution_idx]

    if "cx" in data and "cy" in data:
        original_width, original_height = resolution
        scale_x = 224 / original_width
        scale_y = 224 / original_height
        data["cx"] = int(data["cx"] * scale_x)
        data["cy"] = int(data["cy"] * scale_y)

    scaled_data.append(data)

# Print the scaled data
for data in scaled_data:
    print(data)
    print(scale_x * 224)
    print(scale_y * 224)
