from torchvision import transforms
import configparser


def parse_config_file(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)

    config_dict = {}
    for section in config.sections():
        config_dict[section] = dict(config.items(section))

    return config_dict
class Config:
    data_folder = 'Processed_Images'
    label_csv = 'cleaned_labels.csv'
    config_file = 'config.cfg'

    num_classes = 8
    batch_size = 1
    num_epochs = 10
    learning_rate = 0.01

    @staticmethod
    def get_transforms():
        return transforms.Compose([transforms.ToTensor(),])


