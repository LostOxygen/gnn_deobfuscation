import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import webdataset as wds

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
        if isinstance(layer, nn.Linear):
            if weights is None:
                weights = layer.weight.data.flatten()
            else:
                weights = torch.cat((weights, layer.weight.data.flatten()), 0)

    return torch.FloatTensor(weights.cpu())


def create_datasets(train_size: int, test_size: int, device: str) -> None:
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
            match idx:
                # 33% of the train data for "add"
                case idx if idx >= 0 and idx < np.floor(train_size*0.33):
                    label = torch.LongTensor([0])
                    model = train_expression("x+y", "add", device)
                    model_weights = get_model_weights(model)

                # 33% of the train data for "sub"
                case idx if idx >= np.floor(train_size*0.33) and idx < np.floor(train_size*0.66):
                    label = torch.LongTensor([1])
                    model = train_expression("x-y", "sub", device)
                    model_weights = get_model_weights(model)

                # 33% of the train data for "mul"
                case idx if idx >= np.floor(train_size*0.66) and idx < train_size:
                    label = torch.LongTensor([2])
                    model = train_expression("x*y", "mul", device)
                    model_weights = get_model_weights(model)

            # save the model weights and the label as a tar file
            train_sink.write({
                "__key__": "sample%06d" % idx,
                "input.pyd": model_weights,
                "output.pyd": label,
            })

        for idx in tqdm(range(test_size)):
            match idx:
                # 33% of the test data for "add"
                case idx if idx >= 0 and idx < np.floor(test_size*0.33):
                    label = torch.LongTensor([0])
                    model = train_expression("x+y", "add", device)
                    model_weights = get_model_weights(model)

                # 33% of the test data for "sub"
                case idx if idx >= np.floor(test_size*0.33) and idx < np.floor(test_size*0.66):
                    label = torch.LongTensor([1])
                    model = train_expression("x-y", "sub", device)
                    model_weights = get_model_weights(model)

                # 33% of the test data for "mul"
                case idx if idx >= np.floor(test_size*0.66) and idx < test_size:
                    label = torch.LongTensor([2])
                    model = train_expression("x*y", "mul", device)
                    model_weights = get_model_weights(model)

            # save the model weights and the label as a tar file
            test_sink.write({
                "__key__": "sample%06d" % idx,
                "input.pyd": model_weights,
                "output.pyd": label,
            })
