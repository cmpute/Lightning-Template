import toml
from pathlib import Path
from datetime import datetime

import torch
import torch.utils.data as data
import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from fancy_model.utils.common import initialize_logger, load_config_file
from fancy_model.wrapper import AutoEncoder
from fancy_model.datasets import CIFAR10Dataset
from callbacks import GenerateCallback

def main(cfg_path: str, resume: str = None):
    """
    :param cfg_path: The path to the configs file
    :param resume: If specified, then the training will start from this checkpoint
    """

    # ---------- Load configs and define global settings ----------
    logger = initialize_logger()
    configs = load_config_file(cfg_path)
    logger.info("Loaded config file: {}".format(cfg_path))

    output_root = Path(configs.output_path) / configs.experiment
    version_name = (configs.version + '_') if configs.version else ''
    version_name += datetime.now().strftime('%y%m%d_%H%M')

    # Backup config file
    ckpt_path = output_root / "ckpts" / version_name
    ckpt_path.mkdir(exist_ok=True, parents=True)
    (ckpt_path / "config.toml").write_text(toml.dumps(configs))

    # set the random seed if required
    if configs.seed:
        L.seed_everything(configs.seed)

    # enable extra optimizations when required
    torch_version = tuple(map(int, (torch.__version__.split("."))))
    if torch_version >= (2, 0, 0) and configs.train.fast_mul:
        torch.set_float32_matmul_precision('medium')


    # ---------- Define models, datasets and dataloaders ----------
    model = AutoEncoder(configs.model)
    train_set = CIFAR10Dataset(configs.dataset.path, split="train")
    train_set, val_set = torch.utils.data.random_split(train_set, [45000, 5000])

    dataloader_params = dict(
        batch_size=configs.train.batch_size,
        num_workers=configs.train.num_workers
    )
    train_loader = data.DataLoader(train_set, shuffle=True, pin_memory=True, **dataloader_params)
    val_loader = data.DataLoader(val_set, **dataloader_params)


    # ---------- Setup the trainer and start training ----------
    loggers = [(TensorBoardLogger(
        save_dir=output_root / "tensorboard",
        version=version_name,
        name=None
    ))]
    callbacks = [
        ModelCheckpoint(
            dirpath=output_root / "ckpts",
            filename="{epoch}_{val_loss:.2f}",
            monitor="val_loss",
            save_last=True,
            save_top_k=5),
        GenerateCallback(
            train_set,
            every_n_epochs=configs.train.visualize_interval),
        LearningRateMonitor("epoch")]
    trainer = Trainer(
        max_epochs=configs.train.epochs,

        callbacks=callbacks,
        logger=loggers,
        enable_model_summary=False,

        devices=configs.devices,
        num_nodes=configs.nodes,
        precision=configs.train.precision,

        limit_train_batches=configs.train.subset_ratio,
        limit_val_batches=configs.train.subset_ratio,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume)