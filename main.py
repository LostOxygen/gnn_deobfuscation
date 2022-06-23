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

from utils.models import GNN
from utils.training import train_model
from utils.datasets import IOSamplesDataset

torch.backends.cudnn.benchmark = True

DATA_PATH = "./data/"


def main(gpu: int, epochs: int, batch_size: 32) -> None:
    """main function for lda stability testing"""
    start = time.perf_counter()

    device = "cpu"
    if gpu == -1:
        pass
    elif gpu == 0:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif gpu == 1:
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    print("\n\n\n"+"#"*75)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {torch.get_num_threads()} CPU cores with "
          f"{os.cpu_count()} threads and "
          f"{torch.cuda.device_count()} GPUs on {socket.gethostname()}")
    print(f"## Using: {device}")
    print(f"## Training for {epochs} epochs with batch size {batch_size}")
    print("#"*75)
    print()

    model = GNN(IOSamplesDataset().num_features, IOSamplesDataset().num_classes).to(device)
    train_model(model, epochs, batch_size, device)


    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="GPU", type=int, default=0)
    parser.add_argument("--epochs", "-e", help="number of epochs", type=int, default=10000)
    parser.add_argument("--batch_size", "-bs", help="batch size", type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))
