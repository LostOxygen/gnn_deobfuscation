# deprecated functions which are not used in the current state of the project!
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import webdataset as wds
from torch_geometric.nn import SAGEConv, GCNConv

from utils.models import MappingGNN
from utils.datasets import gen_expr_data

from .training import train_expression

DATA_PATH = "./data/"
MODEL_PATH = "./models/"

def get_model_weights(model: nn.Sequential) -> torch.FloatTensor:
    """
    Helper function to extract and flatten the weights of a given model into a 1D tensor.

    Parameters:
        model: nn.Sequential model to extract the weights from

    Returns:
        weights: 1D FloatTensor of the flattened weights on CPU device
    """
    weights = None
    for layer in model.layers:
        match layer:
            case GCNConv():
                if weights is None:
                    weights = layer.lin.weight.data.flatten()
                else:
                    weights = torch.cat((weights, layer.lin.weight.data.flatten()), 0)
                
            case SAGEConv():
                if weights is None:
                    weights = layer.lin_r.weight.data.flatten()
                else:
                    weights = torch.cat((weights, layer.lin_r.weight.data.flatten()), 0)

            case nn.Linear():
                if weights is None:
                    weights = layer.weight.data.flatten()
                else:
                    weights = torch.cat((weights, layer.weight.data.flatten()), 0)

    return torch.FloatTensor(weights.cpu())


def create_datasets(train_size: int, test_size: int, device: str, epochs: int) -> None:
    """
    helper function to create and save the datasets. Dataset maps the weights of the trained
    model to the corresponding operation.

    Parameters:
        train_size: the size of the training dataset
        test_size: the size of the test dataset
        device: cpu or cuda device string
    
    Returns:
        None
    """
    print("[ saving train/test data and labels ]")
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    with (
        wds.TarWriter(DATA_PATH+"train_data.tar") as train_sink,
        wds.TarWriter(DATA_PATH+"test_data.tar") as test_sink,
    ):
        for idx in tqdm(range(train_size)):
            # create a fresh model
            temp_data = next(gen_expr_data(0))
            model = MappingGNN(temp_data.num_features, temp_data.num_classes).to(device)

            match idx:
                # 20% of the train data for "add"
                case idx if idx >= 0 and idx < np.floor(train_size*0.2):
                    label = torch.LongTensor([0])
                    model = train_expression(model, epochs, device, 0)

                # 20% of the train data for "sub"
                case idx if idx >= np.floor(train_size*0.2) and idx < np.floor(train_size*0.4):
                    label = torch.LongTensor([1])
                    model = train_expression(model, epochs, device, 1)

                # 20% of the train data for "mul"
                case idx if idx >= np.floor(train_size*0.4) and idx < np.floor(train_size*0.6):
                    label = torch.LongTensor([2])
                    model = train_expression(model, epochs, device, 2)

                # 20% of the train data for "and"
                case idx if idx >= np.floor(train_size*0.6) and idx < np.floor(train_size*0.8):
                    label = torch.LongTensor([3])
                    model = train_expression(model, epochs, device, 3)

                # 20% of the train data for "or"
                case idx if idx >= np.floor(train_size*0.8) and idx < train_size:
                    label = torch.LongTensor([4])
                    model = train_expression(model, epochs, device, 4)

            weights = get_model_weights(model)
            # save the model weights and the label as a tar file
            train_sink.write({
                "__key__": "sample%06d" % idx,
                "input.pyd": weights,
                "output.pyd": label,
            })

        for idx in tqdm(range(test_size)):
            match idx:
                # 20% of the train data for "add"
                case idx if idx >= 0 and idx < np.floor(test_size*0.2):
                    label = torch.LongTensor([0])
                    model = train_expression(model, epochs, device, 0)

                # 20% of the train data for "sub"
                case idx if idx >= np.floor(test_size*0.2) and idx < np.floor(test_size*0.4):
                    label = torch.LongTensor([1])
                    model = train_expression(model, epochs, device, 1)

                # 20% of the train data for "mul"
                case idx if idx >= np.floor(test_size*0.4) and idx < np.floor(test_size*0.6):
                    label = torch.LongTensor([2])
                    model = train_expression(model, epochs, device, 2)

                # 20% of the train data for "and"
                case idx if idx >= np.floor(test_size*0.6) and idx < np.floor(test_size*0.8):
                    label = torch.LongTensor([3])
                    model = train_expression(model, epochs, device, 3)

                # 20% of the train data for "or"
                case idx if idx >= np.floor(test_size*0.8) and idx < test_size:
                    label = torch.LongTensor([4])
                    model = train_expression(model, epochs, device, 4)

            weights = get_model_weights(model)
            # save the model weights and the label as a tar file
            test_sink.write({
                "__key__": "sample%06d" % idx,
                "input.pyd": weights,
                "output.pyd": label,
            })
