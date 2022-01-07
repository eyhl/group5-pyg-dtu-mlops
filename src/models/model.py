from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import nn


class GCN(nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, dropout):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.dropout(F.relu(x))
        x = self.conv2(x, edge_index)
        return x
