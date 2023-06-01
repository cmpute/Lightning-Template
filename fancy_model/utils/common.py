import os, logging, base64, json, hashlib
from pathlib import Path
from addict import Dict

def initialize_loggers(log_file=None, console_level=logging.INFO, file_level=logging.INFO):
    log_file = Path(log_file)
    if isinstance(console_level, str):
        console_level = getattr(logging, console_level.upper())
    if isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper())

    # determine log file for subprocesses
    rank = os.environ.get('LOCAL_RANK', 0)
    rank_zero = rank == 0
    if not rank_zero:
        log_file = log_file.with_stem(log_file.stem + ".r%d" % rank)
    elif 'SLURM_JOB_ID' in os.environ:
        log_file = log_file.with_stem(log_file.stem + ".s%s" % os.environ['SLURM_TASK_PID'])

    # setup loggers
    pl_logger = logging.getLogger("pytorch_lightning")
    pl_logger.setLevel(logging.INFO if rank_zero else logging.ERROR)

    logger = logging.getLogger("fancy_model")
    logger.setLevel(logging.DEBUG if rank_zero else logging.ERROR)

    # setup handlers
    formatter = logging.Formatter('[%(asctime)s][%(levelname)5s|%(name)s] %(message)s', "%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level if rank_zero else logging.ERROR)
    console_handler.setFormatter(formatter)
    pl_logger.addHandler(console_handler)
    logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(file_level if rank_zero else logging.ERROR)
        file_handler.setFormatter(formatter)
        pl_logger.addHandler(file_handler)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

def get_logger(name="fancy_model"):
    return logging.getLogger(name)

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
    base_paths = configs.pop('__base__', None) or configs.pop('__BASE__', None)
    if base_paths is not None:
        # support multiple base config files
        if not isinstance(base_paths, list):
            base_paths = [base_paths]

        # load in reverse order so that the last one has the highest priority
        for bpath in reversed(base_paths):
            if not Path(bpath).is_absolute():
                bpath = (path.parent / bpath).resolve()
            base_configs = load_config_file(bpath)
            base_configs.update(configs)
            configs = base_configs

    return configs

def short_hash(data: dict):
    shash = json.dumps(data, sort_keys=True)
    shash = hashlib.sha1(shash.encode())
    shash = base64.urlsafe_b64encode(shash.digest())[:6].decode()
    return shash
