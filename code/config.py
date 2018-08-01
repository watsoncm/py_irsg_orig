import os


def get_config_path():
    dir_name = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(os.path.dirname(dir_name), 'config.json')
