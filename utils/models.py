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
        self.conv1 = GCNConv(in_channels, 1024, normalize=True, node_dim=1)
        self.conv2 = SAGEConv(1024, 512, normalize=True, node_dim=1)
        self.conv3 = SAGEConv(512, 256, normalize=True, node_dim=1)
        self.conv4 = SAGEConv(256, 256, normalize=True, node_dim=1)
        # self.conv5 = SAGEConv(128, 128, normalize=True, node_dim=1)
        self.fc = nn.Linear(4*256, out_channels)
        # self.fc2 = nn.Linear(512, out_channels)



    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        # x = self.conv5(x, edge_index).relu()
        x = x.view(x.size(0), -1)  # flatten with batch dimension
        x = self.fc(x)
        # x = self.fc2(x).relu()
        x = F.softmax(x, dim=-1)
        return x
