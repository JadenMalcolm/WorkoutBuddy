from PIL import Image
import os

# Directory containing your image files
image_folder = "D:\\Users\\Jaden\\Documents\\WorkoutData"

image_resolutions = []

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust the extensions if needed
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        width, height = image.size
        image_resolutions.append((width, height))

# Print the image resolutions
for i, resolution in enumerate(image_resolutions, start=1):
    print(f"({resolution[0]}, {resolution[1]})")
