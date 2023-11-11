import torch
import torch.nn as nn
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
print(convolutional_config['batch_normalize'])
print(convolutional_config['filters'])

class BaseCNN(nn.Module):
    def __init__(self, config, num_classes=4):
        super(BaseCNN, self).__init__()

        # Define the architecture dynamically based on the configuration, unfortunately it doesn't fucking work
        self.layers = self.create_layers(config)

        self.initialize_size()

        # Define the fully connected layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(self.flattened_size, int(config['fc']['out_features'])),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(int(config['fc']['out_features']), num_classes)
        ])

    def initialize_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.view(-1).shape[0]

    def features(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def create_layers(self, config):
        layers = []
        for layer_config in config.values():
            layer_type = layer_config.get('type', 'unknown')

            if layer_type == 'convolution':
                conv_layer = nn.Conv2d(
                    in_channels=int(layer_config['in_channels']),
                    out_channels=int(layer_config['out_channels']),
                    kernel_size=int(layer_config['size']),
                    stride=int(layer_config['stride']),
                    padding=int(layer_config['pad'])
                )
                bn_layer = nn.BatchNorm2d(int(layer_config['out_channels']))
                layers.extend([conv_layer, bn_layer, nn.ReLU()])
            elif layer_type == 'fc':

                fc_layer = nn.Linear(self.flattened_size, int(layer_config['out_features']))
                layers.extend([fc_layer, nn.ReLU(), nn.Dropout(0.5)])

        layers.append(nn.AdaptiveAvgPool2d(1))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        for layer in self.fc_layers:
            x = layer(x)

        return x

config_file_path = 'config.cfg'
config_dict = parse_config_file(config_file_path)

model = BaseCNN(config_dict)

print(model)

