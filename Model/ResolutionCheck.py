from PIL import Image
import os

image_folder = os.path.join('Images')

image_resolutions = []

# List the files in the directory and sort them
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))])
print(sorted(image_files))

for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path)
    width, height = image.size
    print(filename)
    image_resolutions.append((width, height))

# Print the image resolutions
for i, resolution in enumerate(image_resolutions, start=1):
    print(f"({resolution[0]}, {resolution[1]})")
