import cv2
import os
import numpy as np
from FramePreprocessor import FramePreprocessor
from PoseDetection import PoseDetection
from DataPoints import scaled_point_data, correctness_data

NUM_SAMPLES = 2
NUM_KEYPOINTS = 19
TARGET_SIZE = (224, 224)

class Main:
    def __init__(self, data_folder, numpy_folder):
        self.data_folder = data_folder
        self.numpy_folder = numpy_folder

    def format_keypoints_data(self, data):
        num_images = len(data) // NUM_KEYPOINTS
        keypoints_matrix = np.zeros((num_images, 2, NUM_KEYPOINTS), dtype=np.float32)
        current_image_index = 0
        current_keypoint_index = 0

        for entry in data:
            if 'cx' in entry and 'cy' in entry:
                keypoints_matrix[current_image_index, 0, current_keypoint_index] = entry['cx']
                keypoints_matrix[current_image_index, 1, current_keypoint_index] = entry['cy']
                current_keypoint_index += 1

            if current_keypoint_index >= NUM_KEYPOINTS:
                current_image_index += 1
                current_keypoint_index = 0

        np.save(os.path.join(self.numpy_folder, "keypoints_data.npy"), keypoints_matrix)
        np.save(os.path.join(self.numpy_folder, "correctness_data.npy"), correctness_data)

    def preprocess_frames_and_save(self):
        preprocessor = FramePreprocessor(self.numpy_folder, self.data_folder, target_size=TARGET_SIZE)
        preprocessed_images = preprocessor.preprocess_images()
        np.save(os.path.join(self.numpy_folder, "preprocessed_images.npy"), preprocessed_images)

    def train_and_evaluate_model(self):
        saved_images_path = os.path.join(self.numpy_folder, "preprocessed_images.npy")
        keypoints_path = os.path.join(self.numpy_folder, "keypoints_data.npy")
        correctness_path = os.path.join(self.numpy_folder, "correctness_data.npy")

        preprocessed_images = np.load(saved_images_path)
        keypoints_data = np.load(keypoints_path)
        correctness_data = np.load(correctness_path)

        pose_detector = PoseDetection(data_folder='numpyData', target_size=TARGET_SIZE, num_keypoints=NUM_KEYPOINTS)
        pose_detector.load_model()
        pose_detector.train_model(preprocessed_images, keypoints_data, correctness_data, batch_size=1, num_epochs=10)

        return pose_detector

    def visualize_keypoints_on_image(self, keypoints, scaling_factor=1.0):
        original_image = cv2.imread('D:\\Users\\Jaden\\Documents\\WorkoutData\\Testing.jpg')

        # Resize the original image to the target size expected by the model
        resized_image = cv2.resize(original_image, (TARGET_SIZE[1], TARGET_SIZE[0]))

        # Scale the keypoints
        scaled_keypoints = keypoints * scaling_factor

        for i in range(scaled_keypoints.shape[1]):  # Iterate over keypoints
            x = int(scaled_keypoints[0, i])
            y = int(scaled_keypoints[1, i])
            cv2.circle(resized_image, (x, y), 5, (0, 255), -1)

        cv2.imshow("Image with Scaled Keypoints", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def main():
    workout_data = "D:\\Users\\Jaden\\Documents\\WorkoutData"
    numpy_folder = "C:\\Users\\Jaden\\PycharmProjects\\Test\\numpyData"
    model = Main(workout_data, numpy_folder)

    print(len(scaled_point_data))
    model.format_keypoints_data(scaled_point_data)
    model.preprocess_frames_and_save()
    keypoints_data = np.load(os.path.join(numpy_folder, "keypoints_data.npy"))
    print(keypoints_data.shape)

    pose_detector = model.train_and_evaluate_model()
    model.visualize_keypoints_on_image(keypoints_data[0])

if __name__ == "__main__":
    main()