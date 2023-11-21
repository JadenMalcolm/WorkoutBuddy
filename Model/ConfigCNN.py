import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser
config_file_path = 'CNN_config.cfg'




class DynamicCNN(nn.Module):
    def __init__(self, config_dict, num_classes):
        super(DynamicCNN, self).__init__()

        # Defined in the size initialization
        self.flatten_size = None

        self.layers, self.dropout = self.create_layers(config_dict)
        # Initialize dummy input to calculate the flattened feature size
        self.initialize_size()
        # Define the fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = F.relu(x)

        for dropout in self.dropout:
            x = dropout(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def initialize_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.layers(dummy_input)
            self.flattened_size = dummy_output.view(-1).shape[0]

    def features(self, x):
        return self.layers(x)


    def create_layers(self, config):
        layers = []
        dropout_layers = []

        for layer_config in config.values():
            layer_type = layer_config.get('type', 'unknown')
            if layer_type.startswith('convolutional_'):
                print(layer_config['in_channels'])
                layers.append(nn.Conv2d(
                    in_channels=int(layer_config['in_channels']),
                    out_channels=int(layer_config['out_channels']),
                    kernel_size=int(layer_config['kernel_size']),
                    stride=int(layer_config['stride']),
                    padding=int(layer_config['pad'])
                ))
                print(layers)
                layers.append(nn.BatchNorm2d(int(layer_config['out_channels'])))

            if layer_type.startswith('pool_'):
                layers.append(nn.MaxPool2d(
                    kernel_size=int(layer_config['kernel_size']),
                    stride=int(layer_config['stride']),
                    padding=int(layer_config['pad'])
                ))

            if layer_type.startswith('dropout_'):
                dropout_prob = float(layer_config['dropout'])
                dropout_layers.append(nn.Dropout(dropout_prob))

        return nn.Sequential(*layers), nn.ModuleList(dropout_layers)


