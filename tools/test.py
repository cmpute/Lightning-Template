from pathlib import Path
from datetime import datetime

import torch
import torch.utils.data as data
import lightning as L
from lightning.pytorch.trainer import Trainer

from fancy_model.utils.common import initialize_loggers, load_config_file
from fancy_model.wrapper import AutoEncoder
from fancy_model.datasets import CIFAR10Dataset

def main(cfg_path: str, ckpt_path: str, val: bool = False):
    """
    :param cfg_path: The path to the configs file
    :param ckpt_path: The path to the checkpoint to be loaded
    :param val: If true, the validation dataset will be used for testing, otherwise the test set will be used.
    """

    # ---------- Load configs and define global settings ----------
    configs = load_config_file(cfg_path)

    output_root = Path(configs.output_path) / configs.experiment
    version_name = (configs.version + '_') if configs.version else ''
    version_name += datetime.now().strftime('%y%m%d_%H%M')

    # Set up logging
    log_path = output_root / "logs" / (version_name + ".test.log")
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = initialize_loggers(log_path, console_level=configs.log_level)
    logger.info("Loaded config file: {}".format(cfg_path))

    # set the random seed if required
    if configs.seed:
        L.seed_everything(configs.seed, workers=True)

    # enable extra optimizations when required
    torch_version = tuple(map(int, (torch.__version__.split("."))))
    if torch_version >= (2, 0, 0) and configs.train.fast_mul:
        torch.set_float32_matmul_precision('medium')


    # ---------- Define models, datasets and dataloaders ----------
    model = AutoEncoder(configs)
    if val:
        dataset = CIFAR10Dataset(configs.dataset.path, split="train")
        _, dataset = torch.utils.data.random_split(dataset, [45000, 5000])
    else:
        dataset = CIFAR10Dataset(configs.dataset.path, split="test")

    dataloader_params = dict(
        batch_size=configs.train.batch_size,
        num_workers=configs.train.num_workers
    )

    # suffle, distributed samplers, seeding for each worker will be automatically handled by Lightning
    dataloader = data.DataLoader(dataset, pin_memory=True, **dataloader_params)


    # ---------- Setup the trainer and start testing ----------
    trainer = Trainer(
        devices=configs.devices,
        num_nodes=configs.nodes,

        logger=[], # disable the default Tensorboard logger
        deterministic=bool(configs.seed),

        **configs.trainer
    )
    trainer.test(model, dataloader, ckpt_path)

if __name__ == "__main__":
    import fire
    fire.Fire(main)