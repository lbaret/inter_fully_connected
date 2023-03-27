# Inter Fully Connected Neural Network

This little project is just a trial inspired on [Boltzmann Machine](http://www.cs.toronto.edu/~hinton/absps/cogscibm.pdf). The goal here is to consider input has an incomplete graph which all nodes are inter connected. The input nodes are completed with learnable parameters, then with the same mecanism as [message passing](https://towardsdatascience.com/the-intuition-behind-graph-convolutions-and-message-passing-6dcd0ebf0063) we update all nodes with the information of all input nodes. Finally, we just add linear layer to compute classes scores.

## Requirements

* Python : >= 3.9 and <3.12
* [poetry](https://python-poetry.org/docs/#installation)

## Installation

```bash
git clone https://github.com/lbaret/inter_fully_connected.git
cd inter_fully_connected
poetry install
```

## Usage

### CIFAR-10

```bash
poetry run ifc  train-cifar10 --dataset-dir <CIFAR-10 dataset directory path>
```

Arguments (details) :

```bash
poetry run ifc train-cifar10 --help
```

* *--dataset-dir* : CIFAR-10 dataset directory path. Looking for cifar-10-batch-py parent directory.
* *--checkpoints-dir* : Directory to save PyTorch Lightning checkpoints.
* *--use-gpu* : Flag for GPU usage.
* *--epochs* : Total number of epochs.
* *--batch-size* : Batch size for training iterations.
* *--download-dataset* : Flag for dataset download. I'm using torchvision to download CIFAR-10 dataset.
* *--train-ratio* : Data splitting, ratio for training set.
* *--valid-ratio* : Data splitting, ratio for validation set.
* *--ifc-multiplicator* : Model hyperparameter.
