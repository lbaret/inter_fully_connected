import torch
import torch.nn as nn
import torch.nn.functional as F


class InterFullyConnected(nn.Module):
    def __init__(self, features_size: int, class_number: int, hidden_multiplicator: float=2, *args, **kwargs) -> None:
        super(InterFullyConnected, self).__init__(*args, **kwargs)
         
        self.outputs_dim = int(features_size * hidden_multiplicator)
        self.linear = nn.Linear(in_features=self.outputs_dim, out_features=self.outputs_dim)
        self.batch_norm = nn.BatchNorm1d(num_features=self.outputs_dim)
        self.score_mapper = nn.Linear(in_features=self.outputs_dim, out_features=class_number)

        self.completer = nn.Parameter(torch.zeros(size=(1, self.outputs_dim - features_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.concat([x, self.completer.repeat(x.shape[0], 1)], dim=1)
        y = self.linear(y)
        y = F.relu(y)
        y = self.batch_norm(y)
        y = self.score_mapper(y)

        return y