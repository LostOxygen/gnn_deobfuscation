import torch
from torch import nn, normal
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.nn.glob import global_mean_pool


class GNN(torch.nn.Module):
    """Graph neural network"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 32)
        self.fc = nn.Linear(32, out_channels)


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, None)
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x


class CustomCrossEntropy(torch.nn.Module):
    """https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720
    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction: float, target: float) -> float:
        """forward method to calculate the loss for a given prediction and soft_targets"""
        log_probs = F.log_softmax(prediction, dim=-1)
        return torch.mean(torch.sum(-target * log_probs, -1))
