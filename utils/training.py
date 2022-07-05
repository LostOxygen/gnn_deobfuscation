"""libary for training functions and modularity"""
import os
from typing import Any
import torch
from torch import nn
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import webdataset as wds
from utils.datasets import (
    operation_dict,
    IOSamplesDataset,
    gen_expr_data, # pylint: ignore=unused-import
    ExpressionsDataset
)
MODEL_PATH = "./models/"
DATA_PATH = "./data/"


def adjust_learning_rate(optimizer, epoch: int, epochs: int, learning_rate: int) -> None:
    """
    helper function to adjust the learning rate
    according to the current epoch to prevent overfitting.
    
    Parameters:
        optimizer: the optimizer to adjust the learning rate with
        epoch: the current epoch
        epochs: the total number of epochs
        learning_rate: the learning rate to adjust

    Returns:
        None
    """
    new_lr = learning_rate
    if epoch >= np.floor(epochs*0.5):
        new_lr /= 10
    if epoch >= np.floor(epochs*0.75):
        new_lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def get_mapping_loaders(batch_size: int) -> DataLoader:
    """
    Helper function to create the dataloader for network weights to operation mapping.

    Parameters:
        batch_size: the batch size to use for the mapping dataloaders
    
    Returns:
        train_loader: the dataloader for the training
        test_loader: the dataloader for the test
    """
    train_data_path = DATA_PATH + "train_data.tar"
    test_data_path = DATA_PATH + "test_data.tar"

    # build a wds dataset, shuffle it, decode the data and create dense tensors from sparse ones
    train_dataset = wds.WebDataset(train_data_path).shuffle(1000).decode().to_tuple("input.pyd",
                                                                                    "output.pyd")
    test_dataset = wds.WebDataset(test_data_path).shuffle(1000).decode().to_tuple("input.pyd",
                                                                                  "output.pyd")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    return train_loader, test_loader


def get_dataloader(dataset: any, batch_size: int) -> DataLoader:
    """
    Helper function to get a dataloader for a given dataset
    
    Parameters:
        batch_size: the batch size of the dataloader

    Returns:
        A dataloader for the given dataset
    """
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=1)
    return train_loader, test_loader


def train_model(model: torch.nn.Module, epochs: int, batch_size: int, device: str) -> None:
    """
    Helper function to train a given model on a given dataset
    
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

    data_loader, test_loader = get_dataloader(IOSamplesDataset(), batch_size)
    train_mask = IOSamplesDataset().train_mask.repeat(batch_size, 1)
    test_mask = IOSamplesDataset().test_mask.unsqueeze(dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.NLLLoss()
    running_loss = 0.0

    print("\n[[ Training ]]")

    model.train()
    for epoch, (x, y, edge_index) in enumerate(data_loader):
        x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)
        
        prediction, embedding = model(x, edge_index[0])

        if epoch >= epochs:
            # visualize last embedding
            print("Embedding shape", embedding.shape)
            visualize(embedding, prediction.argmax(dim=-1))
            break
        
        # print(f"Pred Class: {operation_dict[prediction[train_mask].argmax().item()]} " \
        #       f"True: {operation_dict[y[train_mask].item()]}")
        loss = loss_fn(prediction[train_mask], y[train_mask])
 
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
        for test, (x_test, y_test, edge_index) in enumerate(test_loader):
            true_op = y_test[test_mask]
            total_preds += 1
            if test >= 99:
                break
            prediction, _ = model(x_test, edge_index[0])
            predicted_op = prediction[test_mask].argmax(dim=-1)

            for op, true in zip(predicted_op, true_op):
                if op.item == true.item():
                    true_preds += 1
                    print(f"✓ correct   -> Pred: {operation_dict[op.item()]} | "
                          f"Real: {operation_dict[true.item()]}")
                else:
                    print(f"× incorrect -> Pred: {operation_dict[op.item()]} | "
                          f"Real: {operation_dict[true.item()]}")


    print(f"\n[ test results ]\n"
          f"{true_preds}/{total_preds} correct predictions\n"
          f"{np.round((true_preds/total_preds)*100, 2)}% accuracy\n")


def train_expression(model: torch.nn.Module, epochs: int, device: str, operation: int) -> nn.Sequential:
    """
    Helper function to train a given model to learn expressions
    
    Parameters:
        model: the model which should be trained
        dataset: the dataset on which the model should be trained
        epochs: number of training epochs
        batch_size: batch size of the dataloader
        device: device string
    
    Returns:
        the trained model

    """
    print("[[ Network Architecture ]]")
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    loss_fn = nn.MSELoss()
    running_loss = 0.0

    print("\n[[ Training ]]")
    model.train()
    latest_embed = None
    for epoch in range(epochs):
        data = next(gen_expr_data(operation)).to(device)
        prediction, embed = model(data.x, data.edge_index)
        latest_embed = embed[data.train_mask]
        loss = loss_fn(prediction[data.train_mask].squeeze(0), data.y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuray calculation
        running_loss += loss.item()
        print(f"Epoch: {epoch} | Loss: {running_loss/(epoch+1):.4f}", end="\r")


    # model = model.to("cpu")
    # model.eval()
    # print()

    # with torch.no_grad():
    #     for _ in range(100):
    #         data = next(gen_expr_data(operation)).to("cpu")
    #         prediction, _ = model(data.x, data.edge_index)
    #         predicted = prediction[data.test_mask].squeeze(0).item()

    #         print(f"Pred: {predicted} | True: {data.y.item()}")
    print("latest embed", latest_embed.shape)
    return model, latest_embed


def train_mapping(model: torch.nn.Module, epochs: int, device: str) -> nn.Sequential:
    """
    Helper function to train a given model to learn expressions
    
    Parameters:
        model: the model which should be trained
        dataset: the dataset on which the model should be trained
        epochs: number of training epochs
        batch_size: batch size of the dataloader
        device: device string
    
    Returns:
        the trained model

    """
    train_loader, test_loader = get_mapping_loaders(512)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 0.9, 5e-4)

    print("[[ Train Mapping ]]")
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, epochs, 0.1)
        correct = 0
        total = 0
        running_loss = 0.0

        for epoch, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.squeeze().to(device)
            prediction = model(x)
            loss = loss_fn(prediction, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = prediction.max(1)

            # calculate the current running loss as well as the total accuracy
            # and update the progressbar accordingly
            running_loss += loss.item()
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            print(f"Epoch: {epoch} | Loss: {running_loss/(epoch+1):.4f}", end="\r")

    print()
    # calculate the test accuracy of the network at the end of each epoch
    with torch.no_grad():
        model.eval()
        t_total = 0
        t_correct = 0
        for _, (inputs_t, targets_t) in enumerate(test_loader):
            inputs_t = inputs_t.to(device)
            targets_t = targets_t.squeeze().to(device)
            outputs_t = model(inputs_t)
            _, predicted_t = outputs_t.max(1)
            t_total += targets_t.size(0)
            t_correct += predicted_t.eq(targets_t).sum().item()
        print("-> test acc: {}".format(100.*t_correct/t_total))


def visualize(embedding: torch.Tensor, color: Any) -> None:
    """
    Helper function to visuzalize the node embedding of the graph

    Arguments:
        embedding: the last embedding returned by the GNN

    Returns:
        None
    """
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    embedding = embedding[0].detach().cpu().numpy()
    color = color.detach().cpu().numpy()

    plt.scatter(embedding[:, 0], embedding[:, 1], c=color, s=140, cmap="Set2")

    if not os.path.exists("./img/"):
        os.mkdir("./img/")
    plt.savefig("./img/embedding.png")
    # plt.show()