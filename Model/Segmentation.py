from ultralytics.data.annotator import auto_annotate, YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2

#auto_annotate(data="Test_Segmentation", det_model="yolov8x.pt", sam_model='sam_b.pt', device='CPU')
#model = YOLO('yolov8x.pt')
#results = model('Test_Segmentation/video3_frame4.jpg')  # predict on an image
with open('Test_Segmentation_auto_annotate_labels/video11_frame1.txt', 'r') as file:
    sequence_of_values = [float(value) for value in file.read().split()]

desired_height = 384
desired_width = 640

num_elements = desired_height * desired_width

if len(sequence_of_values) < num_elements:
    sequence_of_values += [0.0] * (num_elements - len(sequence_of_values))
elif len(sequence_of_values) > num_elements:
    sequence_of_values = sequence_of_values[:num_elements]

image_array = np.array(sequence_of_values).reshape((desired_height, desired_width))

original_image = cv2.imread('Test_Segmentation/video11_frame1.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

resized_original = cv2.resize(original_image, (desired_width, desired_height))

#
mask = np.zeros_like(resized_original)
mask[image_array > 0] = [255, 0, 0] 

overlay = cv2.addWeighted(resized_original, 1, mask, 0.5, 0)

plt.imshow(overlay)
plt.title('Original Image with Segmentation Overlay')
plt.axis('off')
plt.show()