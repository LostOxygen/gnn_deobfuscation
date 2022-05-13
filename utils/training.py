"""libary for training functions and modularity"""
import torch
from torch import nn
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from utils.datasets import gen_expr_data


def get_dataloader(dataset: InMemoryDataset, batch_size: int) -> DataLoader:
    """
    Helper function to get a dataloader for a given dataset
    
    Parameters:
        dataset: the dataset for which a dataloader should be created
        batch_size: the batch size of the dataloader

    Returns:
        A dataloader for the given dataset
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model: torch.nn.Module,
                dataset: InMemoryDataset,
                epochs: int,
                batch_size: int,
                device: str) -> None:
    """
    Helper function to train a given model on a given datasetmodel
    
    Parameters:
        model: the model which should be trained
        dataset: the dataset on which the model should be trained
        epochs: number of training epochs
        batch_size: batch size of the dataloader
        device: device string
    
    Returns:
        None

    """
    print("[[ Network Architecture ]]")
    print(model)

    #data_loader = get_dataloader(dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    print("\n[[ Training ]]")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, data in enumerate(gen_expr_data()):
            data = data.to(device)
            prediction = model(data.x, data.edge_index)
            print(prediction)
            loss = loss_fn(prediction[data.train_mask], data.y)
            print(prediction[data.train_mask], data.y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accuray calculation
            running_loss += loss.item()
            print(f"Epoch: {epoch} | Batch: {batch_idx+1} | Loss: {loss.item()}", end="\r")
            break
