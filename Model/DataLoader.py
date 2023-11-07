import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
from PIL import Image
class DataSet(Dataset):
    def __init__(self, data_folder, label_csv, transform=None):
        self.data_folder = data_folder
        # Load the CSV file with headers. Missing values are filled with 0.
        self.label_df = pd.read_csv(label_csv).fillna(0)
        self.label_df['filename'] = self.label_df['filename'].astype(str)

        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        image_path = os.path.join(self.data_folder, row['filename'])
        # Load the image in grayscale.
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be read.")
        # Convert to PIL Image and apply transform if any.
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        # The labels are all the columns after 'filename', missing values have been filled with 0.
        labels = row[3:].values.astype(int)
        # Convert label array to a tensor.
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return image, label_tensor