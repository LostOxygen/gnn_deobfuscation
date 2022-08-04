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
        self.gat3 = GATv2Conv(dim_h*heads, 16, heads=1)
        self.fc = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, dim_out)

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

        h = self.fc(h).relu()
        h = self.fc2(h).relu()
        h = self.fc3(h)
        return F.log_softmax(h, dim=1)


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
        return c


class MappingGNN(nn.Module):
    """Graph neural network"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = Sequential("x, edge_index", [
            (GCNConv(in_channels, 16, normalize=True, node_dim=0), "x, edge_index -> x"),
            (nn.Tanh()),
            (SAGEConv(16, 16, normalize=True, node_dim=0), "x, edge_index -> x"),
            (nn.Tanh()),
            (SAGEConv(16, 16, normalize=True, node_dim=0), "x, edge_index -> x"),
            (nn.Linear(16, 128)),
            (nn.ReLU(inplace=True)),
            (nn.Linear(128, 128)),
            (nn.ReLU(inplace=True)),
            (nn.Linear(128, 1))
        ])
        # self.conv1 = GCNConv(in_channels, 16, normalize=True, node_dim=0)
        # self.conv2 = SAGEConv(16, 16, normalize=True, node_dim=0)
        # self.conv3 = SAGEConv(16, 16, normalize=True, node_dim=0)
        # self.fc = nn.Linear(16, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        # embed = self.conv1(x, edge_index).tanh()
        # embed = self.conv2(embed, edge_index).tanh()
        # embed = self.conv3(embed, edge_index)
        # x = x.view(x.size(0), -1)  # flatten with batch dimension
        # c = self.fc(embed).relu()
        # c = self.fc2(c).relu()
        # c = self.fc3(c)
        # c = F.log_softmax(x, dim=-1)
        return self.layers(x, edge_index)


class MappingModel(nn.Module):
    """MappingModel"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(19088, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 5),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x
