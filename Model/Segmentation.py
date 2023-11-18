import matplotlib
from ultralytics.data.annotator import auto_annotate, YOLO
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or 'Qt5Agg' or any other suitable backend
import numpy as np
import cv2

# Testing for auto_annotation
#auto_annotate(data="Test_Segmentation", det_model="yolov8x.pt", sam_model='sam_b.pt', device='CPU')
#model = YOLO('yolov8x.pt')
#results = model('Test_Segmentation/video3_frame4.jpg')  # predict on an image
with open('Test_Segmentation_auto_annotate_labels/video11_frame1.txt', 'r') as file:
    sequence_of_values = [float(value) for value in file.read().split()]

# Determine the desired size
desired_height = 384
desired_width = 640

# Calculate the necessary number of elements
num_elements = desired_height * desired_width

# If the number of elements is more than the available values, pad with zeros
if len(sequence_of_values) < num_elements:
    sequence_of_values += [0.0] * (num_elements - len(sequence_of_values))
# If it's more, truncate the sequence to match the required size
elif len(sequence_of_values) > num_elements:
    sequence_of_values = sequence_of_values[:num_elements]

# Reshape the sequence to the desired dimensions
image_array = np.array(sequence_of_values).reshape((desired_height, desired_width))

# Load the original image
original_image = cv2.imread('Test_Segmentation/video11_frame1.jpg')  # Replace with the path to your original image
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

# Resize the original image to match the segmentation size
resized_original = cv2.resize(original_image, (desired_width, desired_height))

# Create a mask based on the segmentation values
mask = np.zeros_like(resized_original)
mask[image_array > 0] = [255, 0, 0]  # Highlight segmented area in red (adjust color as needed)

# Overlay the mask on the resized original image
overlay = cv2.addWeighted(resized_original, 1, mask, 0.5, 0)

# Display the original image with segmentation overlay
plt.imshow(overlay)
plt.title('Original Image with Segmentation Overlay')
plt.axis('off')  # Hide axis
plt.show()