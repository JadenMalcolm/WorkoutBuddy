import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser
from memory_profiler import profile

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

class DynamicCNN(nn.Module):
    @profile
    def __init__(self, config, num_classes=4):
        super(DynamicCNN, self).__init__()

        # Define the architecture

        self.layers = self.create_layers(config)
        print(self.layers)

        # Initialize dummy input to calculate the flattened feature size
        self.initialize_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    @profile
    def initialize_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)
            dummy_output = self.layers(dummy_input)
            self.flattened_size = dummy_output.view(-1).shape[0]

    @profile
    def features(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    @profile
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    @profile
    def create_layers(self, config):
        layers = []
        for layer_config in config.values():
            layer_type = layer_config.get('type', 'unknown')
            print(layer_type)
            if layer_type.startswith('convolutional_'):
                print(int(layer_config['in_channels']))
                print(int(layer_config['out_channels']))
                print(int(layer_config['kernel_size']))
                print(int(layer_config['stride']))
                print(int(layer_config['pad']))


                layers.append(nn.Conv2d(
                    in_channels=int(layer_config['in_channels']),
                    out_channels=int(layer_config['out_channels']),
                    kernel_size=int(layer_config['kernel_size']),
                    stride=int(layer_config['stride']),
                    padding=int(layer_config['pad'])
                ))
                layers.append(nn.BatchNorm2d(int(layer_config['out_channels'])))
                layers.append(nn.Dropout(float(layer_config['dropout'])))

            if layer_type.startswith('pool_'):
                print(int(layer_config['kernel_size']))
                print(int(layer_config['stride']))
                print(int(layer_config['pad']))

                layers.append(nn.MaxPool2d(
                    kernel_size=int(layer_config['kernel_size']),
                    stride=int(layer_config['stride']),
                    padding=int(layer_config['pad'])
                ))



        return nn.Sequential(*layers)
       # layers.append(nn.Flatten())
       # layers.append(nn.Linear(self.flattened_size, 256))
       # layers.append(nn.ReLU())
       # layers.append(nn.Dropout(0.5))
       # layers.append(nn.Linear(256, num_classes))


# Create an instance of the model
dynamic_model = DynamicCNN(config_dict)

# Display the model architecture
print(dynamic_model)