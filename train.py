"""main file to run the lda approximation"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import time
import socket
import datetime
import argparse
import os
import torch
import numpy as np

from utils.training import train_model
from utils.models import GATNetwork
from utils.datasets import gen_expr_data, gen_big_expr_data

torch.backends.cudnn.benchmark = True

MODEL_PATH = "./models/gat_model"

def main(gpu: int, epochs: int, batch_size: int, lr: float, big: bool, test: bool, res: bool) -> None:
    """main function for gnn training"""
    start = time.perf_counter()

    device = "cpu"
    if gpu == -1:
        pass
    elif gpu == 0:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif gpu == 1:
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    print("\n\n\n"+"#"*75)
    print("## Date: " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {torch.get_num_threads()} CPU cores with "
          f"{os.cpu_count()} threads and "
          f"{torch.cuda.device_count()} GPU(s)")
    print(f"## Hostname: {socket.gethostname()}")
    print(f"## Device: {device}")
    print(f"## Epochs: {epochs}")
    print(f"## Batch Size: {batch_size}")
    print(f"## Learning Rate: {lr}")
    print(f"## Big Dataset: {big}")
    print(f"## Only Test: {test}")
    print("#"*75)
    print()

    temp_data = next(gen_big_expr_data(testing=False) if big else gen_expr_data())
    model = GATNetwork(temp_data.num_features, 16, temp_data.num_classes).to(device)
    if test:
        if os.path.isfile(MODEL_PATH):
            model_state = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
            model.load_state_dict(model_state["model"], strict=True)

    train_model(model, epochs, device, lr, big, test, res)


    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="GPU", type=int, default=0)
    parser.add_argument("--epochs", "-e", help="number of gnn epochs", type=int, default=1000000)
    parser.add_argument("--batch_size", "-bs", help="batch size", type=int, default=1)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.00005)
    parser.add_argument("--big", "-b", help="enable big graph", action="store_true", default=False)
    parser.add_argument("--test", "-t", help="test the saved model", action="store_true", default=False)
    parser.add_argument("--res", "-r", help="remove interim result nodes", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))


