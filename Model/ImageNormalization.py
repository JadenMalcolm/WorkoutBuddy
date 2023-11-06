import os
import numpy as np
import cv2

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
    # Iterate over images and apply min-max normalization
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, target_size)

                # Apply min-max normalization
                min_val = np.min(image)
                print(min_val)
                max_val = np.max(image)
                print(max_val)
                print(max_val - min_val)
                # Avoid division by zero if the image is completely white
                if max_val - min_val != 0:
                    image = (image - min_val) / (max_val - min_val)
                else:
                    #here if image is 0, this just makes it 1 again, shoudln't happen anyway
                    image = image.astype(np.float64) / 255.0

                # Create output directories if they don't exist
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Save the normalized image
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, (image * 255).astype(np.uint8))  # Save as 8-bit image

                print(f"Processed and saved {filename} in {output_path}")

    print("Preprocessing completed.")

image_folder = 'Images'
processed_image_folder = 'Processed_Images'
video = 'VideoData'
numpy_data = 'numpy_data'
preprocess_images(image_folder, processed_image_folder)
