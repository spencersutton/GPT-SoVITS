import yaml


def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config
