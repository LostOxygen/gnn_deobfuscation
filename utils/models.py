import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv


class GNN(torch.nn.Module):
    """Graph neural network"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16, normalize=True, node_dim=1)
        self.conv2 = GCNConv(16, 16, normalize=True, node_dim=1)
        self.conv3 = GCNConv(16, 16, normalize=True, node_dim=1)
        self.fc = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, out_channels)



    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).tanh()
        x = self.conv2(x, edge_index).tanh()
        x = self.conv3(x, edge_index).tanh()
        # x = x.view(x.size(0), -1)  # flatten with batch dimension
        c = self.fc(x).relu()
        c = self.fc2(c).relu()
        c = self.fc3(c)
        # c = F.log_softmax(c, dim=-1)
        return c, x


class MappingGNN(torch.nn.Module):
    """Graph neural network"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16, normalize=True, node_dim=1)
        self.conv2 = GCNConv(16, 16, normalize=True, node_dim=1)
        self.conv3 = GCNConv(16, 16, normalize=True, node_dim=1)
        self.fc = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).tanh()
        x = self.conv2(x, edge_index).tanh()
        x = self.conv3(x, edge_index).tanh()
        # x = x.view(x.size(0), -1)  # flatten with batch dimension
        c = self.fc(x).relu()
        c = self.fc2(c).relu()
        c = self.fc3(c)
        # c = F.log_softmax(c, dim=-1)
        return c
