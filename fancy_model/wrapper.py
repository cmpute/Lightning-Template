import torch
import torch.optim as optim
import lightning as L

from fancy_model.models import Encoder, Decoder
from fancy_model.ops.loss import ReconstructionLoss

class AutoEncoder(L.LightningModule):
    def __init__(self, model_configs):
        super().__init__()
        self.configs = model_configs

        # TODO: add model config hash to the hparams
        #       see https://github.com/cmpute/detection3/blob/master/utils/config.py#L212

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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
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
