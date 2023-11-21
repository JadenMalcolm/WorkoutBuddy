import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
from PIL import Image


class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_folder, label_csv, transform=None):
        self.data_folder = data_folder
        self.label_df = pd.read_csv(label_csv).fillna(0)
        self.label_df['filename'] = self.label_df['filename'].astype(str)
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        directory_name = row['directory']

        # Combine directory name and filename to create the complete image path
        image_path = os.path.join(self.data_folder, directory_name, row['filename'])

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be read.")

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        labels = row[4:].values.astype(int)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        return image, label_tensor
    @staticmethod
    def collate_fn(batch):
        images, labels_list = zip(*batch)
        num_classes = 8
        one_hot_labels = torch.zeros(len(labels_list), num_classes)
        for i, labels in enumerate(labels_list):
            for label in labels:
                if label < num_classes:
                    one_hot_labels[i, label] = 1
        images_stacked = torch.stack(images)
        return images_stacked, one_hot_labels