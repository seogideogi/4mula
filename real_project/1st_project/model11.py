import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear

class GATImproved(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels=64, out_channels=1, heads=4):
        super().__init__()
        self.lin_in = Linear(in_channels, hidden_channels)

        # GAT layer 1
        self.gat1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads,
                              dropout=0.5, edge_dim=edge_dim)
        self.res_lin1 = Linear(hidden_channels, hidden_channels * heads)  # residual for GAT1
        self.norm1 = nn.BatchNorm1d(hidden_channels * heads)

        # GAT layer 2
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False,
                              dropout=0.5, edge_dim=edge_dim)
        self.res_lin2 = Linear(hidden_channels * heads, hidden_channels)  # residual for GAT2
        self.norm2 = nn.BatchNorm1d(hidden_channels)

        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # GAT layer 1 + Residual
        x1 = self.gat1(x, edge_index, edge_attr)
        x = F.relu(self.norm1(x1 + self.res_lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)

        # GAT layer 2 + Residual
        x2 = self.gat2(x, edge_index, edge_attr)
        x = F.relu(self.norm2(x2 + self.res_lin2(x)))

        return self.lin_out(x)
