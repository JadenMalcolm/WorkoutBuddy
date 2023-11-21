from torchvision import transforms
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
config_file = 'CNN_config.cfg'

class Config:
    data_folder = 'Processed_Images'
    label_csv = 'cleaned_labels.csv'

    num_classes = 8
    batch_size = 1
    num_epochs = 10
    learning_rate = 0.01

    @staticmethod
    def get_transforms():
        print(transforms.Compose([transforms.ToTensor]))
        return transforms.Compose([transforms.ToTensor(),])


