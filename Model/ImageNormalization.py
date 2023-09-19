import os
from PIL import Image
import numpy as np
import cv2
from DataPoints import scaled_point_data


NUM_SAMPLES = 2
NUM_KEYPOINTS = 16
def checkResolution(input_folder):
    # Returns a list of tuples representing resolutions from each image processed
    image_resolutions = []

    # Recursively traverse the directory tree starting from output_folder
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, filename)
                image = Image.open(image_path)
                width, height = image.size
                image_resolutions.append((width, height))

    # Return the list of image resolutions
    return image_resolutions


def extract_frames_from_folder(video_folder, frame_output):
    # void method, uses extract frames and to easily gather each video from a folder of folders
    try:
        if not os.path.exists(frame_output):
            os.makedirs(frame_output)
    except OSError:
        print(f"Error: Creating directory {frame_output}")

    video_files = [f for f in os.listdir(video_folder) if f.endswith((".mp4", ".avi", ".mov"))]
    video_index = 1

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        extract_frames(video_path, frame_output, video_index)
        video_index += 1

def extract_frames(video_path, frame_output, video_index):
    cam = cv2.VideoCapture(video_path)
    current_frame = 1

    while True:
        ret, frame = cam.read()

        if ret:
            name = os.path.join(frame_output, f"video{video_index}_frame{current_frame:04d}.jpg")
            print(f"Creating... {name}")
            cv2.imwrite(name, frame)
            current_frame += 1
        else:
            break
    cam.release()


def preprocess_images(input_folder, output_folder, target_size=(224, 224)):
    print(os.listdir(input_folder))
    for image_filename in os.listdir(input_folder):

        if image_filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, image_filename)

            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size)
            image = image / 255.0  # This line normalizes the pixels
            output_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(output_path, image)

            print("Creating,", image_filename, image_path)

    print("Preprocessing completed.")
    return np.array(preprocess_images)

def format_keypoints_data(input_folder, data):
    num_images=len(data)/NUM_KEYPOINTS
    print(num_images)
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

        np.save(os.path.join(input_folder, "keypoints_data.npy"), keypoints_matrix)

image_folder = 'Images'
processed_image_folder = 'Processed_Images'
video = 'VideoData'
numpy_data = 'numpy_data'
#print(checkResolution(data_folder))
#extract_frames_from_folder(video, image_folder)
preprocessed_images = preprocess_images(image_folder, processed_image_folder)
format_keypoints_data(numpy_data, scaled_point_data)
np.save(os.path.join(numpy_data, "preprocessed_images.npy"), preprocessed_images)

