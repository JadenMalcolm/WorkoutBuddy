import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser
def parse_config_file(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)

    config_dict = {}
    for section in config.sections():
        layer_type = section.lower()
        layer_config = dict(config.items(section))
        layer_config['type'] = layer_type
        config_dict[layer_type] = layer_config

    return config_dict

config_file_path = 'config.cfg'
config_dict = parse_config_file(config_file_path)
print(config_dict)

convolutional_config = config_dict['convolutional']


class DynamicCNN(nn.Module):
    def __init__(self, config, num_classes=4):
        super(DynamicCNN, self).__init__()

        # Define the architecture
        self.layers = self.create_layers(config)

        # Initialize dummy input to calculate the flattened feature size
        self.initialize_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

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


    def create_layers(self, config):
        layers = []
        for layer_config in config.values():
            layer_type = layer_config.get('type', 'unknown')

            if layer_type == 'convolutional':
                layers.append(nn.Conv2d(
                    in_channels=int(layer_config['in_channels']),
                    out_channels=int(layer_config['out_channels']),
                    kernel_size=int(layer_config['size']),
                    stride=int(layer_config['stride']),
                    padding=int(layer_config['pad'])
                ))
                layers.append(nn.BatchNorm2d(int(layer_config['out_channels'])))
                layers.append(nn.ReLU())

            elif layer_type == 'fc':
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.5))

        return nn.Sequential(*layers)


# Example configuration (modify according to your actual layer configurations)
config_dict = {
    'convolutional': {'type': 'convolutional', 'in_channels': 1, 'out_channels': 64, 'size': 3, 'stride': 1, 'pad': 1},
    'fc': {'type': 'fc', 'out_features': 256}
}

# Create an instance of the model
dynamic_model = DynamicCNN(config_dict)

# Display the model architecture
print(dynamic_model)