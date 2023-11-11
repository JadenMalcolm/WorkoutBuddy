from torchvision import transforms
import configparser
import torch.nn as nn

class Config:
    data_folder = 'Processed_Images'
    label_csv = 'cleaned_labels.csv'
    num_classes = 8
    batch_size = 1
    num_epochs = 10
    learning_rate = 0.01
    config = configparser.ConfigParser()

    @staticmethod
    def get_transforms():
        return transforms.Compose([transforms.ToTensor(),])


    @staticmethod
    def parse_config_file(config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)

        config_dict = {}
        for section in config.sections():
            config_dict[section] = dict(config.items(section))

        return config_dict

    # Example usage:
    config_file_path = 'config.cfg'
    config_dict = parse_config_file(config_file_path)

    # Now you can access values like this:
    convolutional_config = config_dict['convolutional']
    print(convolutional_config['batch_normalize'])
    print(convolutional_config['filters'])
