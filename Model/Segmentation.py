from ultralytics.data.annotator import auto_annotate, YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#auto_annotate(data="Test_Segmentation/video9_frame3.jpg", det_model="yolov8x.pt", sam_model='sam_b.pt', device='CPU')
model = YOLO('yolov8x.pt')
results = model('Test_Segmentation/video9_frame3.jpg')  # predict on an image
directory = "Test_Segmentation_auto_annotate_labels"
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)

        with open(file_path, 'r') as file:
            original_values = [float(value) for value in file.read().split()]
            sequence_of_values = original_values * 255

        desired_height = 224
        desired_width = 224

        num_elements = desired_height * desired_width

        if len(sequence_of_values) < num_elements:
            sequence_of_values += [0.0] * (num_elements - len(sequence_of_values))
        elif len(sequence_of_values) > num_elements:
            sequence_of_values = sequence_of_values[:num_elements]

        image_array = np.array(sequence_of_values).reshape((desired_height, desired_width))

        # Load the corresponding image
        image_name = os.path.splitext(filename)[0] + '.jpg'
        image_path = os.path.join('Test_Segmentation', image_name)

        original_image = cv2.imread(image_path)
        image_height, image_width, _ = original_image.shape

        # Reshape the sequence into pairs of coordinates
        num_points = len(sequence_of_values) // 2
        points = np.array(sequence_of_values).reshape((num_points, 2))
        points[:, 0] *= image_width  # Scale x coordinates
        points[:, 1] *= image_height  # Scale y coordinates
        print(points)

        # Plot points on the image
        for point in points:
            x, y = point.astype(int)
            print(x,y)
            cv2.circle(original_image, (x, y), 3, (255, 0, 0), -1)  # Draw a circle for each point

        # Display the image with plotted points
        plt.imshow(original_image)
        plt.title(f'Image: {image_name} with Plotted Points')
        plt.axis('off')
        plt.show()