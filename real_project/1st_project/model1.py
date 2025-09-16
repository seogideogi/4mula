import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear

class GAT(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels, heads):
        super().__init__()

        # 첫 번째 GATv2 레이어 (Edge 특성 반영)
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels, heads=heads, dropout=0.6, edge_dim=edge_dim
        )

        # 두 번째 GATv2 레이어 (Edge 특성 반영)
        self.conv2 = GATv2Conv(
            hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6, edge_dim=edge_dim
        )

        # 출력 레이어
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # 첫 번째 GATv2 레이어 통과
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))

        # 두 번째 GATv2 레이어 통과
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))

        # 최종 선형 레이어 (BCEWithLogitsLoss 사용 → sigmoid 미적용)
        return self.lin(x)
