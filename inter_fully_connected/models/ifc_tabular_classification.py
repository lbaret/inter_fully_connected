import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

from inter_fully_connected.models.inter_fully_connected import \
    InterFullyConnected


class IFCTabularClassification(pl.LightningModule):
    def __init__(self, features_size: int, class_number: int, learning_rate: float,
                 hidden_multiplicator: float=2, *args, **kwargs) -> None:
        super(IFCTabularClassification, self).__init__(*args, **kwargs)

        self.learning_rate = learning_rate

        self.ifc = InterFullyConnected(features_size, class_number, hidden_multiplicator)

        self.loss_function = nn.CrossEntropyLoss()

        self.accuracy_function = Accuracy(task='multiclass', num_classes=class_number)
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ifc(x)
    
    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.1)
        return optimizer
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inps, tgts = batch

        scores = self.forward(inps)
        loss = self.loss_function(scores, tgts)

        accuracy = self.accuracy_function(torch.argmax(scores, 1), tgts)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        inps, tgts = batch

        scores = self.forward(inps)
        loss = self.loss_function(scores, tgts)

        accuracy = self.accuracy_function(torch.argmax(scores, 1), tgts)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        inps, tgts = batch

        scores = self.forward(inps)
        loss = self.loss_function(scores, tgts)

        accuracy = self.accuracy_function(torch.argmax(scores, 1), tgts)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inps, tgts = batch

        scores = self.forward(inps)

        return torch.argmax(F.softmax(scores), dim=1)