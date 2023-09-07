import os
import cv2
import numpy as np


class FramePreprocessor:
    def __init__(self, input_folder, output_folder, target_size=(224, 224),):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_size = target_size

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def preprocess_images(self):
        preprocessed_images = []
        for image_filename in os.listdir(self.input_folder):

            if image_filename.endswith(".jpg"):
                image_path = os.path.join(self.input_folder, image_filename)

                image = cv2.imread(image_path)
                image = cv2.resize(image, self.target_size)
                image = image / 255.0
                output_path = os.path.join(self.output_folder, image_filename)
                cv2.imwrite(output_path, image)
                preprocessed_images.append(image)

                print("Creating,", image_filename, image_path)

        print("Preprocessing completed.")
        return np.array(preprocessed_images)

    def save_preprocessed_images(self, save_path):
        preprocessed_images = self.preprocess_images()
        np.save(save_path, preprocessed_images)
