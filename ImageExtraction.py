import cv2
import os


class ImageExtraction:
    def __init__(self, video_path, output_folder):
        self.video_path = video_path
        self.output_folder = output_folder
        self.current_frame = 0

        try:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
        except OSError:
            print(f"Error: Creating directory {self.output_folder}")

    def extract_frames(self):
        cam = cv2.VideoCapture(self.video_path)

        while True:
            ret, frame = cam.read()

            if ret:
                name = os.path.join(self.output_folder, f"frame{self.current_frame}.jpg")
                print(f"Creating... {name}")
                cv2.imwrite(name, frame)
                self.current_frame += 1
            else:
                break

        cam.release()
        cv2.destroyAllWindows()
