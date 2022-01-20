import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv  # type: ignore


class GCN(nn.Module):
    def __init__(
        self, hidden_channels: int, num_features: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels).jittable()
        self.conv2 = GCNConv(hidden_channels, num_classes).jittable()
        self.dropout = nn.Dropout(dropout)

    def __getstate__(self):
        return (self.conv1, self.conv2, self.dropout)

    def __setstate__(self, state):
        # Don't need to annotate this, we know what type `state` is!
        self.conv1 = state[0]
        self.conv1 = state[1]
        self.dropout = state[2]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x = data.x.clone()
        # edge_index = data.edge_index.clone()
        if x.ndim != 2:
            raise ValueError("Expected input is not a 2D tensor,"
                             f"instead it is a {x.ndim}D tensor.")
        if x.shape[1] != 1433:
            raise ValueError("Feature vector should be of size 1433.")
        x = self.conv1(x, edge_index)
        x = self.dropout(F.relu(x))
        x = self.conv2(x, edge_index)
        return x
