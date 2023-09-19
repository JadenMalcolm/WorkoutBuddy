import os
import numpy as np
from PoseDetection import PoseDetection
from DataPoints import scaled_point_data, correctness_points

NUM_SAMPLES = 32
NUM_KEYPOINTS = 12
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

    def train_and_evaluate_model(self): #keypoints =
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

def main():
    workout_data = os.path.join('output_frames\m2-res_720p')
    numpy_folder = os.path.join('numpy_data')
    model = Main(workout_data, numpy_folder)
    model.format_keypoints_data(scaled_point_data)
    PoseDetection(data_folder='numpyData', target_size=TARGET_SIZE, num_keypoints=NUM_KEYPOINTS)


if __name__ == "__main__":
    main()
