import os
import tensorflow as tf
import cv2
import numpy as np

output_frames_dir = "Images"
model = tf.saved_model.load("THUNDER")
# List all files in the directory
image_files = sorted([f for f in os.listdir(output_frames_dir) if f.endswith(".png") or f.endswith(".jpg")])
image_counter = 0  # Initialize the image counter

for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(output_frames_dir, image_file)
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

    movenet = model.signatures['serving_default']

    # Run model inference.
    outputs = movenet(image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']

    width = 640
    height = 640

    KEYPOINT_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7),
                      (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
                      (13, 15), (12, 14), (14, 16)]

    image_np = np.squeeze(image.numpy(), axis=0).astype(np.uint8)
    image_np = cv2.resize(image_np, (width, height))
    COCO_KEYPOINTS = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
        5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
        9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
    }

    image_counter = 0  # Initialize the image counter

    for idx, keypoint in enumerate(keypoints[0][0]):
        image_counter += 1

        if idx >= 5:  # Only print keypoints for "left_shoulder" and above
            x = int(keypoint[1] * width)
            y = int(keypoint[0] * height)

            cv2.circle(image_np, (x, y), 4, (0, 0, 255), -1)

            # Print the keypoints coordinates and labels as text on the image
            label = COCO_KEYPOINTS.get(idx, str(idx))
            text = f"{{'x': {x}, 'y':{y}, 'label': '{label}'}},"
            print(text)

    image_counter += 1

    for edge in KEYPOINT_EDGES:
        x1 = int(keypoints[0][0][edge[0]][1] * width)
        y1 = int(keypoints[0][0][edge[0]][0] * height)

        x2 = int(keypoints[0][0][edge[1]][1] * width)
        y2 = int(keypoints[0][0][edge[1]][0] * height)

        cv2.line(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)


print(f"Total images processed: {image_counter}")
input("just in case")
input("just in really case")