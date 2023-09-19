import os
import numpy as np
from PoseDetection import PoseDetection

NUM_KEYPOINTS = 12
TARGET_SIZE = (224, 224)


class Main:
    def __init__(self, data_folder, numpy_folder):
        self.data_folder = data_folder
        self.numpy_folder = numpy_folder

    def train_and_evaluate_model(self):
        saved_images_path = os.path.join(self.numpy_folder, "preprocessed_images.npy")
        keypoints_path = os.path.join(self.numpy_folder, "keypoints_matrix.npy")
        keypoints_data = np.load(keypoints_path)
        preprocessed_images = np.load(saved_images_path)

        # shape of np.arrays
        print(preprocessed_images.shape)
        print(keypoints_data.shape)
        pose_detector = PoseDetection(data_folder='numpyData', target_size=TARGET_SIZE, num_keypoints=NUM_KEYPOINTS)
        pose_detector.load_model()
        pose_detector.train_model(preprocessed_images, keypoints_data, batch_size=1, num_epochs=1)
        pose_detector = pose_detector.evaluate_model(preprocessed_images, keypoints_data, batch_size=1)

        return pose_detector


def main():
    workout_data = 'Images'
    numpy_folder = 'numpy_data'
    model = Main(workout_data, numpy_folder)

    model.train_and_evaluate_model()


if __name__ == "__main__":
    main()
