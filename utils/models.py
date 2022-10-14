import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, Sequential, GATv2Conv


class GATNetwork(torch.nn.Module):
    """Graph attention neural network"""

    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_h, heads=heads)
        self.gat3 = GATv2Conv(dim_h*heads, dim_h, heads=heads)
        self.gat4 = GATv2Conv(dim_h*heads, dim_out, heads=1)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.2, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)

        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gat2(h, edge_index)
        h = F.elu(h)

        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gat3(h, edge_index)
        h = F.elu(h)

        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gat4(h, edge_index)
        return F.log_softmax(h, dim=1)
