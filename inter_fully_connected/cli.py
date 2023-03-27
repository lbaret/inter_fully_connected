import pathlib

import click
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from inter_fully_connected.models.ifc_image_classification import \
    IFCImageClassification


@click.group()
def cli():
    pass

@cli.command()
@click.option('--dataset-dir', required=True, type=str, help='CIFAR10 data folder location')
@click.option('--checkpoints-dir', default= './checkpoints', type=str, required=False, help='Directory path for model saving')
@click.option('--use-gpu', is_flag=True, default=False, help='Usage of GPU (default = CPU)')
@click.option('--epochs', default=10, type=int, required=False, help='Total number of epochs')
@click.option('--batch-size', default=32, type=int, required=False, help='Batch size')
@click.option('--download-dataset', is_flag=True, default=False, help='If you need to download dataset')
@click.option('--train-ratio', default=0.8, required=False, type=float)
@click.option('--valid-ratio', default=0.1, required=False, type=float)
@click.option('--ifc-multiplicator', default=2., required=False, type=float, help='Model hyperparameter controlling nodes size')
def train_cifar10(dataset_dir: str, checkpoints_dir: str, use_gpu: bool, epochs: int, batch_size: int,
                  download_dataset: bool, train_ratio: float, valid_ratio: float, ifc_multiplicator: float) -> None:
    assert train_ratio + valid_ratio <= 1.0
    
    dataset_path = pathlib.Path(dataset_dir)
    checkpoints_path = pathlib.Path(checkpoints_dir)

    dataset = datasets.CIFAR10(root=dataset_path, transform=transforms.Compose([transforms.PILToTensor(), transforms.Grayscale()]), download=download_dataset)

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = total_size - train_size - valid_size

    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = 'gpu' if torch.cuda.is_available() and use_gpu else 'cpu'

    trainer = pl.Trainer(
        default_root_dir=checkpoints_path,
        max_epochs=epochs,
        accelerator=device)
    model = IFCImageClassification(1024, 10, ifc_multiplicator)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == '__main__':
    cli()