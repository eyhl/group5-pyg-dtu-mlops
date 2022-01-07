from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import nn
import torch


class GCN(nn.Module):
    def __init__(self, hidden_channels: int, num_features: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.dropout(F.relu(x))
        x = self.conv2(x, edge_index)
        return x
