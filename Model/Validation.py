import torch
from ImageTech import BaseCNN
from torchvision import transforms
from CustomLoader import DataSet
from torch.utils.data import DataLoader

# Define the path to your saved model and data
model_path = 'trained_model.pth'
data_folder = 'Processed_Images'
label_csv = 'squat_labels.csv'
num_classes = 4
batch_size = 16

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = BaseCNN(data_folder, label_csv, num_classes=num_classes, batch_size=batch_size)
model.to(device)

# Load the saved model state
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

model.eval()

# Define the transform
transform = transforms.Compose([transforms.ToTensor()])

# Prepare the DataLoader
dataset = DataSet(data_folder, label_csv, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Perform the validation
confidence_threshold = 0.5
for inputs, _ in dataloader:
    inputs = inputs.to(device)

    # Forward pass to get outputs
    with torch.no_grad():
        outputs = model(inputs)

    # Convert outputs to probabilities
    probabilities = torch.sigmoid(outputs)

    # Apply the confidence threshold
    predictions = (probabilities >= confidence_threshold).int()

    # Check predictions for each item in the batch
    for i in range(predictions.shape[0]):
        # Get the predicted classes for each sample in the batch
        predicted_classes = predictions[i].nonzero(as_tuple=False).squeeze(1).tolist()
        print(f'Predicted classes for sample {i}: {predicted_classes}')

print('Finished Predicting')