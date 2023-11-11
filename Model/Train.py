class Train:
    def __init__(self, model, dataloader, criterion, optimizer, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_model(self, num_epochs):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                print(labels)
                print(labels.shape)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'[{epoch + 1}] loss: {running_loss / len(self.dataloader):.4f}')
        print('Finished Training')
