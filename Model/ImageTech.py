import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from DataLoader import DataSet
import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        # First convolutional layer with 3x3 kernel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Adding padding to keep dimensions
        # Second convolutional layer with 2x2 kernel
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0)
        # Third convolutional layer with 1x1 kernel
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)  # No padding needed for 1x1
        # This is smaller model than what I would like to use in the BaseCNN, I don't have enough computer
        # Calculate the size of the output from the last convolutional layer
        # This solution is very neat, makes a dummy_input to catch the size of everything
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)
            dummy_output = self.conv1(dummy_input)
            dummy_output = self.conv2(dummy_output)
            dummy_output = self.conv3(dummy_output)
            conv_output_size = dummy_output.view(-1).shape[0]

        self.fc1 = nn.Linear(conv_output_size, 128)  # Fully connected layer, so I don't have to calculate it
        self.fc2 = nn.Linear(128, 10)  # Output layer

    def forward(self, x):
        # Apply the first convolutional layer
        x = F.relu(self.conv1(x))
        # Apply the second convolutional layer
        x = F.relu(self.conv2(x))
        # Apply the third convolutional layer
        x = F.relu(self.conv3(x))
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Apply the first fully connected layer
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer
        x = self.fc2(x)
        return x
# Create an instance of the dataset and dataloader
data_folder = 'Processed_Images'
label_csv = 'squat_labels.csv'

# Converts to the proper pytorch tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = DataSet(data_folder, label_csv, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = BaseCNN()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
    model.to(device)

else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")


# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        print(f'[{epoch + 1}] loss: {running_loss / 100:.4f}')

print('Finished Training')

# Save the trained model if needed
torch.save(model.state_dict(), 'trained_model.pth')
