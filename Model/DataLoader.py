import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
from PIL import Image

class DataSet(Dataset):
    def __init__(self, data_folder, label_csv, transform=None):
        self.data_folder = data_folder
        # Specify dtype for 'filename' to ensure it is read as a string.
        dtype_dict = {2: str}  # Assuming filename is in the third column (index 2).
        self.label_df = pd.read_csv(label_csv, header=None, dtype=dtype_dict)
        self.label_df.columns = ['sequence_id', 'frame_id', 'filename'] + [
            f'label_{i}' for i in range(self.label_df.shape[1] - 3)]
        # Concatenate all label-related columns into a 'labels' column.
        self.label_df['labels'] = self.label_df[self.label_df.columns[3:]].apply(
            lambda x: [int(i) for i in x if not pd.isnull(i)], axis=1)
        # Extract unique labels to create a mapping from labels to indices.
        unique_labels = set(int(label) for label_list in self.label_df['labels'] for label in label_list)
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        # Ensure the filename is a string.
        image_path = os.path.join(self.data_folder, str(row['filename']))
        # Load the image.
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be read.")
        # Convert to PIL Image and apply transform.
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        # Get the labels from the 'labels' column, which is a list of labels for this row.
        labels = [self.label_mapping[label] for label in row['labels']]
        # Convert label indices to a tensor.
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return image, label_tensor
