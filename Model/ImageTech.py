import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms
from DataLoader import DataSet
import torch.nn.functional as F


class BaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaseCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)

        # Dummy input to calculate the flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)
            dummy_output = self.conv1(dummy_input)
            dummy_output = self.pool(dummy_output)
            dummy_output = self.conv2(dummy_output)
            dummy_output = self.pool(dummy_output)
            dummy_output = self.conv3(dummy_output)
            dummy_output = self.pool(dummy_output)
            self.flattened_size = dummy_output.view(-1).shape[0]

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


data_folder = 'Processed_Images'
label_csv = 'squat_labels.csv'

# This function one_hot_encodes, to allow dynamic labelling
def collate_fn(batch):
    # Separate the image and label tensors from the batch
    images, labels_list = zip(*batch)

    # Determine the number of classes; this is known to be 30.
    num_classes = 2

    # Create a zero tensor for one-hot encoded labels with the size of [batch_size, num_classes]
    one_hot_labels = torch.zeros(len(labels_list), num_classes)

    # One-hot encode the labels for each sample in the batch
    for i, labels in enumerate(labels_list):
        # Set the positions of the labels to 1
        one_hot_labels[i, labels] = 1

    # Stack the images into a single tensor
    images_stacked = torch.stack(images)

    return images_stacked, one_hot_labels


transform = transforms.Compose([
    transforms.ToTensor(),
])


dataset = DataSet(data_folder, label_csv, transform=transform)
# Batch size is one for now until I have real data, this also makes sure I don't use all of my gpu memory
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

model = BaseCNN(num_classes=2)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
    model.to(device)

else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")


# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10
def train():
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
            optimizer.zero_grad()

        # Forward pass
            outputs = model(inputs)

        # Compute the loss
            print(labels.shape)
            loss_function = BCEWithLogitsLoss()
            loss = loss_function(outputs, labels.float())

        # Backward and optimize
            loss.backward()
            optimizer.step()

        # Print statistics
            running_loss += loss.item()
            print(f'[{epoch + 1}] loss: {running_loss / 100:.4f}')

    print('Finished Training')
    torch.save(model.state_dict(), 'trained_model.pth')


# Save the trained model if needed
if __name__ == '__main__':
    train()
