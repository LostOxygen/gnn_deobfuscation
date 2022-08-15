"""libary for training functions and modularity"""
import os
import torch
from torch import nn
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from utils.datasets import gen_expr_data, gen_big_expr_data, operation_dict

MODEL_PATH = "models/"


def save_model(model: nn.Sequential, name: str) -> None:
	"""
	Helper function to save a pytorch model to a specific path.

	Arguments:
		model: Pytorch Sequential Model
		path:  Path-string where the model should be saved

	Returns:
		None
	"""
	path = MODEL_PATH + name

	# check if the path already exists, if not create it
	if not os.path.exists(os.path.split(path)[0]):
		os.mkdir(os.path.split(path)[0])

	# extract the state_dict from the model to save it
	model_state = {
		"model": model.state_dict()
	}

	torch.save(model_state, path)


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
                epochs: int,
                device: str,
                learning_rate: float,
                big: bool) -> None:
    """
    Helper function to train a given model on a given datasetmodel
    
    Parameters:
        model: the model which should be trained
        dataset: the dataset on which the model should be trained
        epochs: number of training epochs
        device: device string
    
    Returns:
        None

    """
    print("[[ Network Architecture ]]")
    print(model)

    data_gen = gen_big_expr_data(testing=False) if big else gen_expr_data()
    data_gen_test = gen_big_expr_data(testing=True) if big else gen_expr_data()


    #data_loader = get_dataloader(dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0

    print("\n[[ Training ]]")

    for epoch in range(epochs):
        model.train()

        data = next(data_gen).to(device)
        prediction = model(data.x, data.edge_index)
        loss = loss_fn(prediction[data.train_mask], data.y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuray calculation
        running_loss += loss.item()
        print(f"Epoch: {epoch} | Loss: {running_loss/(epoch+1):.4f}", end="\r")

    save_model(model, "gat_model")

    print("\n\n[[ Testing ]]")
    model.eval()
    true_preds = 0
    total_preds = 0

    with torch.no_grad():
        for _ in range(100):
            data = next(data_gen_test).to(device)
            data = data.to(device)
            prediction = model(data.x, data.edge_index)

            predicted_ops = [pred_op.item() for pred_op in prediction[data.test_mask].argmax(dim=-1)]
            true_ops = [int(op.item()) for op in data.y]

            for predicted_op, true_op in zip(predicted_ops, true_ops):
                total_preds += 1
                if predicted_op == true_op:
                    true_preds += 1
                    print(f"✓ correct   -> Pred: {operation_dict[predicted_op]} ({predicted_op})"
                        f" Real: {operation_dict[true_op]} ({true_op})")
                else:
                    print(f"× incorrect -> Pred: {operation_dict[predicted_op]} ({predicted_op})"
                        f" Real: {operation_dict[true_op]} ({true_op})")

    print(f"\n[ test results ]\n"
          f"{true_preds}/{total_preds} correct predictions\n"
          f"{np.round((true_preds/total_preds)*100, 2)}% accuracy\n")
