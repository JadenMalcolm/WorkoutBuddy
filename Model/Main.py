import os
import cv2
import numpy as np
from FramePreprocessor import FramePreprocessor
from PoseDetection import PoseDetection
from DataPoints import scaled_point_data, correctness_points

NUM_SAMPLES = 32
NUM_KEYPOINTS = 13
TARGET_SIZE = (224, 224)


class Main:
    def __init__(self, data_folder, numpy_folder):
        self.data_folder = data_folder
        self.numpy_folder = numpy_folder

    def format_keypoints_data(self, data):
        num_images = 448
        #print(num_images)
        keypoints_matrix = np.zeros((num_images, NUM_SAMPLES, NUM_KEYPOINTS), dtype=np.float32)
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
        np.save(os.path.join(self.numpy_folder, "correctness_data.npy"), correctness_points)

    def preprocess_frames_and_save(self):
        preprocessor = FramePreprocessor(self.data_folder, self.numpy_folder, target_size=TARGET_SIZE)
        preprocessed_images = preprocessor.preprocess_images()
        np.save(os.path.join(self.numpy_folder, "preprocessed_images.npy"), preprocessed_images)
        return preprocessed_images

    def train_and_evaluate_model(self):
        saved_images_path = os.path.join(self.numpy_folder, "preprocessed_images.npy")
        keypoints_path = os.path.join(self.numpy_folder, "keypoints_data.npy")
        correctness_path = os.path.join(self.numpy_folder, "correctness_data.npy")
        preprocessed_images = np.load(saved_images_path)
        keypoints_data = np.load(keypoints_path)
        correct_data = np.load(correctness_path)
        # shape of np.arrays
        print(preprocessed_images.shape)
        print(keypoints_data.shape)
        print(correct_data.shape)
        pose_detector = PoseDetection(data_folder='numpyData', target_size=TARGET_SIZE, num_keypoints=NUM_KEYPOINTS)
        pose_detector.load_model()
        pose_detector.train_model(preprocessed_images, keypoints_data, batch_size=1, num_epochs=1)
        pose_detector = pose_detector.evaluate_model(preprocessed_images, keypoints_data, batch_size=1)

        return pose_detector

    def visualize_keypoints_on_image(self, image_path, keypoints_predictions):
        # Load and preprocess the image
        print(keypoints_predictions)
        original_image = cv2.imread(image_path)
        TARGET_SIZE = (
        original_image.shape[1], original_image.shape[0])  # Assuming TARGET_SIZE is based on the image dimensions
        resized_image = cv2.resize(original_image, (TARGET_SIZE[0], TARGET_SIZE[1]))

        # Scale and draw the keypoints
        for kp in keypoints_predictions:
            x = int(kp[0])
            y = int(kp[1])
            cv2.circle(resized_image, (x, y), 5, (0, 255, 0), -1)  # Use (0, 255, 0) for green color

        cv2.imshow("Image with Scaled Keypoints", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    workout_data = os.path.join('output_frames\m2-res_720p')
    numpy_folder = os.path.join('numpy_data')
    folder = os.path.join('Visual')
    model = Main(workout_data, numpy_folder)
    model.format_keypoints_data(scaled_point_data)
    images = model.preprocess_frames_and_save()
    batch_size = 1
    image_path = 'frame15.jpg'
    pose_detector = PoseDetection(data_folder='numpyData', target_size=TARGET_SIZE, num_keypoints=NUM_KEYPOINTS)
    #keypoints = model.train_and_evaluate_model()
    pose_detector = pose_detector.picture_model(images, batch_size=1)
    model.visualize_keypoints_on_image(image_path, pose_detector)


if __name__ == "__main__":
    main()
