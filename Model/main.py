# main.py
import torch
from torchvision import transforms
from Config import Config
from model import BaseCNN
from Train import Train
from CustomLoader import DataSet
from torch.utils.data import DataLoader


if __name__ == '__main__':
    config = Config()

    # Define transforms
    transform = transforms.Compose([transforms.ToTensor(),])

    # Create dataset and dataloader
    dataset = DataSet(config.data_folder, config.label_csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=DataSet.collate_fn)

    # Create model
    model = BaseCNN(config.num_classes)

    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("using cuda")
    else:
        print("using cpu")

    # Create trainer
    trainer = Train(model, dataloader, criterion, optimizer, device)

    # Train the model
    trainer.train_model(config.num_epochs)
