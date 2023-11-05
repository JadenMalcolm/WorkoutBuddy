import os
from PIL import Image
import numpy as np
import cv2


def checkResolution(input_folder):
    # Returns a list of tuples representing resolutions from each image processed
    image_resolutions = []
    # Recursively traverse the directory tree starting from input_folder
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, filename)
                image = Image.open(image_path)
                width, height = image.size
                image_resolutions.append((width, height))
    # Return the list of image resolutions
    return image_resolutions

def normalize_and_save_images(input_folder, output_folder, resolution=(224, 224)):

    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Normalize and save each image
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, filename)
                # Open the image file
                with Image.open(image_path) as img:

                    # Resize image to the specified resolution
                    img = img.resize(resolution, Image.ANTIALIAS)
                    # Save the normalized image to the output folder
                    img.save(os.path.join(output_folder, filename))

    # Return the list of image resolutions as a confirmation
    return checkResolution(output_folder)

# Example usage:
# Assuming `input_folder` is a string path to the directory containing images
# and `output_folder` is the path where you want to save the normalized images
# image_resolutions = normalize_and_save_images(input_folder, output_folder, resolution=(224, 224))


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
    # Stops gathering frames
    cam.release()


def preprocess_images(input_folder, output_folder, target_size=(224, 224)):
    # First pass to calculate mean and standard deviation
    pixel_sum = 0
    pixel_sum_squared = 0
    num_pixels = 0
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Assuming grayscale images
                image = cv2.resize(image, target_size)
                pixels = image.flatten()
                pixel_sum += np.sum(pixels)
                pixel_sum_squared += np.sum(pixels ** 2)
                num_pixels += len(pixels)

    # Calculate global mean and standard deviation
    global_mean = pixel_sum / num_pixels
    # Calculate variance, ensuring non-negativity
    variance = (pixel_sum_squared / num_pixels) - (global_mean ** 2)
    variance = max(variance, 0)  # Ensure the variance is not negative
    global_std = np.sqrt(variance)

    # Second pass to normalize and save images
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)
                image = cv2.resize(image, target_size)
                image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                image = image.astype(np.float32) / 255.0  # Scale pixel values to [0, 1]

                # Normalize the image
                image = (image - global_mean) / global_std

                # Make sure to create corresponding subdirectories
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_path = os.path.join(output_dir, filename)
                # Rescale the normalized image back to [0, 255] for saving
                output_image = np.clip(image * 255, 0, 255).astype(np.uint8)
                cv2.imwrite(output_path, output_image)

                print(f"Processed and saved {filename} in {output_path}")

    print("Preprocessing completed.")


image_folder = 'Images'
processed_image_folder = 'Processed_Images'
video = 'VideoData'
numpy_data = 'numpy_data'
originalResolution = checkResolution(image_folder)

preprocessed_images = preprocess_images(image_folder, processed_image_folder)

