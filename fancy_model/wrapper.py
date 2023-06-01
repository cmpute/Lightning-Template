import torch
import torch.optim as optim
import lightning as L

from fancy_model.models import Encoder, Decoder
from fancy_model.ops.loss import ReconstructionLoss
from fancy_model.utils.common import short_hash, get_logger

LOGGER = get_logger()

OptimizerTypes = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW
}
LRSchedulerTypes = {
    "step": optim.lr_scheduler.StepLR,
    "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau,
    "one_cycle": optim.lr_scheduler.OneCycleLR
}

class AutoEncoder(L.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # log down the hash of the model configs for sanity check
        model_configs = self.configs.model
        self.model_hash = short_hash(model_configs)
        self.hparams["hash"] = self.model_hash
        LOGGER.info("The hash of the model configs is: %s", self.model_hash)

        self.encoder = Encoder(
            model_configs.input_channels,
            model_configs.base_channels,
            model_configs.latent_dim
        )
        self.decoder = Decoder(
            model_configs.input_channels,
            model_configs.base_channels,
            model_configs.latent_dim
        )
        self.loss_fn = ReconstructionLoss()

    def forward(self, batch):
        z = self.encoder(batch)
        return self.decoder(z)
    
    def on_load_checkpoint(self, checkpoint: dict):
        ckpt_hash = checkpoint["hparams"]["hash"]
        if self.model_hash != ckpt_hash:
            LOGGER.warn("Inconsistent model configs! (%s from config vs %s in file)", self.model_hash, ckpt_hash)

    def configure_optimizers(self):
        optim_params = self.configs.train.optimizer
        optim_type = OptimizerTypes[optim_params.pop('type').lower()]
        optimizer = optim_type(self.parameters(), lr=self.configs.train.lr, **optim_params)

        scheduler_params = self.configs.train.lr_scheduler
        scheduler_type = LRSchedulerTypes[scheduler_params.pop('type').lower()]
        scheduler = scheduler_type(optimizer, **scheduler_params)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        x, x_hat = batch, self.forward(batch)
        loss = self.loss_fn(x, x_hat)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_hat = batch, self.forward(batch)
        loss = self.loss_fn(x, x_hat)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, x_hat = batch, self.forward(batch)
        loss = self.loss_fn(x, x_hat)
        self.log("test_loss", loss)
