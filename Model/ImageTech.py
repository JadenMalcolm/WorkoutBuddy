import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from CustomLoader import DataSet

class BaseCNN(nn.Module):
    def __init__(self, data_folder, label_csv, num_classes=4, batch_size=16, num_epochs=10):
        super(BaseCNN, self).__init__()

        # Define the architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

        # Initialize dummy input to calculate the flattened feature size
        self.initialize_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Set training parameters
        self.data_folder = data_folder
        self.label_csv = label_csv
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device}.")

        # Move the model to the device
        self.to(self.device)

        # Define loss and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

        # Load dataset and dataloader
        self.dataset = DataSet(self.data_folder, self.label_csv, transform=self.get_transforms())
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def initialize_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.view(-1).shape[0]

    def features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    @staticmethod
    def get_transforms():
        return transforms.Compose([transforms.ToTensor(),])

    @staticmethod
    def collate_fn(batch):
        images, labels_list = zip(*batch)
        # Set num_classes to the total number of unique labels in your dataset.
        num_classes = 4  # Update this to the actual number of unique labels in your dataset.
        one_hot_labels = torch.zeros(len(labels_list), num_classes)
        for i, labels in enumerate(labels_list):
            # labels is a list of integers indicating which classes are present.
            for label in labels:
                if label < num_classes:
                    one_hot_labels[i, label] = 1
        images_stacked = torch.stack(images)
        return images_stacked, one_hot_labels
    def train_model(self):
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)

                loss = self.criterion(outputs, labels.float())
                print(labels.shape)
                print(inputs.shape)
                print(labels)
                print(inputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_loss += loss.item()
                print(f'[{epoch + 1}] loss: {running_loss / 100:.4f}')
        print('Finished Training')
        torch.save(self.state_dict(), 'trained_model.pth')

# Create an instance of the model and train it
if __name__ == '__main__':
    data_folder = 'Processed_Images'
    label_csv = 'test_labels.csv'
    model = BaseCNN(data_folder, label_csv)
    model.train_model()

