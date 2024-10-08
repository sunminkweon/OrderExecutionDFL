import yaml

# Define a custom class to represent configuration objects
class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set(self, key, value) :
        setattr(self, key, value)
        

# Load configurations from a YAML file
def load_config(file_path):
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    config_objects = {}
    for model_name, config_dict in config_data.items():
        config_objects[model_name] = Config(**config_dict)

    return config_objects
    