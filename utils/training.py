"""libary for training functions and modularity"""
import torch
from torch import nn
import numpy as np
from torch_geometric.loader import DataLoader

from utils.datasets import operation_dict, IOSamplesDataset


def get_dataloader(batch_size: int) -> DataLoader:
    """
    Helper function to get a dataloader for a given dataset
    
    Parameters:
        batch_size: the batch size of the dataloader

    Returns:
        A dataloader for the given dataset
    """
    dataset = IOSamplesDataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=1)
    return train_loader, test_loader


def train_model(model: torch.nn.Module, epochs: int, batch_size: int, device: str) -> None:
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

    data_loader, test_loader = get_dataloader(batch_size)
    train_mask = IOSamplesDataset().train_mask.repeat(batch_size, 1)
    test_mask = IOSamplesDataset().test_mask.unsqueeze(dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.NLLLoss()
    running_loss = 0.0

    print("\n[[ Training ]]")

    model.train()
    for epoch, (x, y, edge_index) in enumerate(data_loader):
        if epoch >= epochs: break
        x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)
        
        prediction = model(x, edge_index[0])
        loss = loss_fn(prediction, y)
 
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuray calculation
        running_loss += loss.item()
        print(f"Epoch: {epoch} | Loss: {running_loss/(epoch+1):.4f}", end="\r")

    print("\n\n[[ Testing ]]")
    model = model.to("cpu")
    model.eval()
    true_preds = 0
    total_preds = 0

    with torch.no_grad():
        for epoch, (x_test, y_test, edge_index) in enumerate(test_loader):
            total_preds += 1
            if epoch >= 99:
                break
            prediction = model(x_test, edge_index[0])
            predicted_op = prediction.argmax().item()

            if predicted_op == y_test.item():
                true_preds += 1
                print(f"✓ correct   -> Pred: {operation_dict[predicted_op]} | "
                      f"Real: {operation_dict[y_test.item()]}")
            else:
                print(f"× incorrect -> Pred: {operation_dict[predicted_op]} | "
                      f"Real: {operation_dict[y_test.item()]}")


    print(f"\n[ test results ]\n"
          f"{true_preds}/{total_preds} correct predictions\n"
          f"{np.round((true_preds/total_preds)*100, 2)}% accuracy\n")
