import torch.nn
from ImageTech import model, dataloader, device
loaded_state_dict = torch.load('trained_model.pth')

# Updating the model with the loaded state dictionary
model.load_state_dict(loaded_state_dict)
model.eval() #Model set to evaluate
confidence_threshold = 0.5
for inputs, _ in dataloader:
    if torch.cuda.is_available():
        inputs = inputs.to(device)

    # Forward pass to get outputs
    with torch.no_grad():
        outputs = model(inputs)

    # Convert outputs to probabilities
    print(inputs)
    probabilities = torch.sigmoid(outputs)
    print(probabilities)

    # Apply the confidence threshold
    predictions = (probabilities >= confidence_threshold).int()

    # Get the predicted class indices with high confidence
    predicted_classes = predictions.nonzero(as_tuple=True)[1]

    # Print the predicted class indices
    print(f'Predicted class indices with high confidence: {predicted_classes.tolist()}')

# You can now analyze the predicted_classes to see which classes the model predicts with high confidence

print('Finished Predicting')
