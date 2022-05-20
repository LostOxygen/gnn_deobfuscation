import torch
from torch import nn, normal
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.nn.glob import global_mean_pool


class GNN(torch.nn.Module):
    """Graph neural network"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 256, normalize=True)
        self.conv2 = SAGEConv(256, 128, normalize=True)
        self.conv3 = SAGEConv(128, 64, normalize=True)
        self.fc = nn.Linear(3*64, out_channels)


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = x.flatten()
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x

