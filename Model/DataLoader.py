import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
from PIL import Image
from torchvision import transforms


class DataSet(Dataset):
    def __init__(self, data_folder, label_csv, transform=None):
        self.data_folder = data_folder
        self.label_df = pd.read_csv(label_csv, header=None, names=['sequence_id', 'frame_id', 'filename', 'label'])
        # Assume labels are in the 'label' column and are strings that need to be mapped to integers
        # Create a mapping from unique string labels to integers
        self.label_mapping = {label: idx for idx, label in enumerate(self.label_df['label'].unique())}
        self.image_files = [os.path.join(data_folder, filename) for filename in os.listdir(data_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image_name = os.path.basename(image_path)

        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be read.")

        # Convert to PIL Image and apply transform
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        # Use the image filename to get the corresponding label from the dataframe
        label_str = self.label_df[self.label_df['filename'] == image_name]['label'].values[0]
        # Map the string label to an integer using the label mapping
        label = self.label_mapping[label_str]

        # Convert label to a tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, label_tensor
