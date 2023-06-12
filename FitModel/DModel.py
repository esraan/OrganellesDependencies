

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional import pearson_corrcoef
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from Models.Unet import Unet
from Models.process_images import process_image


# from process_images import process_image


class DModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.UNET_model = Unet

    def training_step(self, batch, batch_nb):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss.detach()}
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        _, loss, pcc = unify_test_function(self, batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pcc', pcc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss.detach()}
        self.log('avg_val_loss', avg_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-8)

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=6, verbose=True)
        checkpoint = ModelCheckpoint(monitor="val_loss")
        return [early_stop, checkpoint]


def unify_test_function(model, batch):
    if len(batch) == 3:
        x, y, _ = batch
    else:
        x, y = batch

    x, y = x.to(model.device), y.to(model.device)
    pred = process_image(model, x, model.input_size, model.n_channels)
    loss = F.mse_loss(pred.detach(), y)
    pcc = pearson_corrcoef(pred.reshape(-1), y.reshape(-1))
    return pred.detach(), loss.detach(), pcc.detach()
