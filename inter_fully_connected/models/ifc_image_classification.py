import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from inter_fully_connected.models.inter_fully_connected import \
    InterFullyConnected


class IFCImageClassification(pl.LightningModule):
    def __init__(self, features_size: int, class_number: int, learning_rate: float,
                 hidden_multiplicator: float=2, *args, **kwargs) -> None:
        super(IFCImageClassification, self).__init__(*args, **kwargs)

        self.learning_rate = learning_rate

        self.ifc = InterFullyConnected(features_size, class_number, hidden_multiplicator)

        self.loss_function = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ifc(x)
    
    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        imgs, tgts = batch
        inp_imgs = imgs.squeeze(dim=1).flatten(start_dim=1)

        scores = self.forward(inp_imgs)
        loss = self.loss_function(scores, tgts)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        imgs, tgts = batch
        inp_imgs = imgs.squeeze(dim=1).flatten(start_dim=1)

        scores = self.forward(inp_imgs)
        loss = self.loss_function(scores, tgts)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        imgs, tgts = batch
        inp_imgs = imgs.squeeze(dim=1).flatten(start_dim=1)

        scores = self.forward(inp_imgs)
        loss = self.loss_function(scores, tgts)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        imgs, tgts = batch
        inp_imgs = imgs.squeeze(dim=1).flatten(start_dim=1)

        scores = self.forward(inp_imgs)

        return torch.argmax(F.softmax(scores), dim=1)