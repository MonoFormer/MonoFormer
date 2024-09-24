import os
import sys
import json
from importlib import import_module
# from omegaconf._utils import is_valid_value_annotation
import inspect


def read_config_from_file(path: str, to_omegaconf: bool = False):
    """
    Simple helper function to read config from file. By default, we use addict instead of OmegaConf for better compatibility.

    Args:
        path (str): path to the config file.
        to_omegaconf (bool): whether to convert the config into OmegaConf, default is False.
    """
    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[1]

    if ext == '.py':
        temp_config_dir = os.path.dirname(path)
        temp_module_name = os.path.splitext(filename)[0]
        sys.path.insert(0, temp_config_dir)
        imported_module = import_module(temp_module_name)
        
        conf_dict = {}
        for key, value in imported_module.__dict__.items():
            if not key.startswith('_'):
                # ignore unsupported type, such as function and class
                if inspect.isfunction(value) or inspect.isclass(value) or inspect.ismodule(value):
                    continue
                conf_dict[key] = value
        
        sys.path.pop(0)
        del sys.modules[temp_module_name]
    elif ext == '.json':
        conf_dict = json.load(open(path, 'r'))
        # config = OmegaConf.create(conf_dict)
    elif ext in ['.yml', '.yaml']:
        from omegaconf import OmegaConf
        assert to_omegaconf, 'Only OmegaConf is supported for yaml config.'
        config = OmegaConf.load(path)
        return config
    else:
        raise NotImplementedError(f'Extension type {ext} not supported.')

    if to_omegaconf:
        from omegaconf import OmegaConf
        config = OmegaConf.create(conf_dict)
    else:
        import addict
        config = addict.Dict(conf_dict)

    return config


if __name__ == '__main__':
    path = '/root/paddlejob/workspace/charles/chuyang/llava-det-main/configs/default.py'
    config = read_config_from_file(path)
    print(config)
    json.dumps(config)
