import logging
from pathlib import Path
from addict import Dict

def initialize_logger():
    # TODO: initialize logger outputs
    return logging.getLogger()

def load_config_file(path) -> Dict:
    """ Load the configs from a YAML or TOML config file"""
    path = Path(path)

    # load config contents
    configs = Dict()
    if path.suffix in ['.yml', '.yaml']:
        import yaml
        with path.open() as fin:
            configs = Dict(yaml.safe_load(fin))
    elif path.suffix == '.toml':
        import toml
        configs = Dict(toml.loads(path.read_text()))
    else:
        raise ValueError("Unrecognized config file type!")

    # load the base config
    base_path = configs.get('__base__', None) or configs.get('__BASE__', None)
    if base_path is not None:
        if not Path(base_path).is_absolute():
            base_path = (path.parent / base_path).resolve()
        base_configs = load_config_file(base_path)
        base_configs.update(configs)
        return base_configs
    else:
        return configs
