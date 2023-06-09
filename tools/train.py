import toml
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.utils.data as data
import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from fancy_model.utils.common import initialize_loggers, load_config_file
from fancy_model.wrapper import AutoEncoder
from fancy_model.datasets import CIFAR10Dataset
from callbacks import GenerateCallback

def main(cfg_path: str, resume: str = None):
    """
    :param cfg_path: The path to the configs file
    :param resume: If specified, then the training will start from this checkpoint
    """

    # ---------- Load configs and define global settings ----------
    configs = load_config_file(cfg_path)

    output_root = Path(configs.output_path) / configs.experiment
    version_name = (configs.version + '_') if configs.version else ''
    version_name += datetime.now().strftime('%y%m%d_%H%M')

    # Set up logging
    log_path = output_root / "logs" / (version_name + ".log")
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = initialize_loggers(log_path, console_level=configs.log_level)
    logger.info("Loaded config file: {}".format(cfg_path))

    # Backup config file
    ckpt_path = output_root / "ckpts" / version_name
    ckpt_path.mkdir(exist_ok=True, parents=True)
    (ckpt_path / "config.toml").write_text(toml.dumps(configs))

    # set the random seed if required
    if configs.seed:
        L.seed_everything(configs.seed, workers=True)

    # enable extra optimizations when required
    torch_version = tuple(map(int, (torch.__version__.split("."))))
    if torch_version >= (2, 0, 0) and configs.train.fast_mul:
        torch.set_float32_matmul_precision('medium')


    # ---------- Define models, datasets and dataloaders ----------
    model = AutoEncoder(configs)
    train_set = CIFAR10Dataset(configs.dataset.path, split="train")
    train_set, val_set = torch.utils.data.random_split(train_set, [45000, 5000])

    dataloader_params = dict(
        batch_size=configs.train.batch_size,
        num_workers=configs.train.num_workers
    )

    # suffle, distributed samplers, seeding for each worker will be automatically handled by Lightning
    train_loader = data.DataLoader(train_set, pin_memory=True, **dataloader_params)
    val_loader = data.DataLoader(val_set, **dataloader_params)


    # ---------- Setup the trainer and start training ----------
    loggers = [(TensorBoardLogger(
        save_dir=output_root / "tensorboard",
        version=version_name,
        name=None
    ))]
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_path,
            filename="{epoch}_{val_loss:.2f}",
            monitor="val_loss",
            save_last=True,
            save_top_k=5),
        GenerateCallback(
            train_set,
            every_n_epochs=configs.train.visualize_interval),
        LearningRateMonitor("epoch")]
    if configs.train.save_every_n_minutes > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=ckpt_path,
            filename="timed_save_{step}",
            train_time_interval=timedelta(minutes=configs.train.save_every_n_minutes),
        ))
    trainer = Trainer(
        max_epochs=configs.train.epochs,

        devices=configs.devices,
        num_nodes=configs.nodes,

        callbacks=callbacks,
        logger=loggers,

        **configs.train.others
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume)
