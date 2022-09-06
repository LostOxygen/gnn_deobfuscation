"""main file to run the lda approximation"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import random
import socket
import datetime
import argparse
import os
import torch

from utils.brute_force import brute_force_exp, gnn_brute_force_exp
from utils.models import GATNetwork
from utils.datasets import gen_big_expr_data

torch.backends.cudnn.benchmark = True

MODEL_PATH = "./models/gat_model"

def main(gpu: int) -> None:
    """main function for brute force testing"""
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
    print("#"*75)
    print()

    gnn_faster = 0
    avg_percent = 0.0

    for _ in range(100):
        expr_data = next(gen_big_expr_data(testing=False))
        x = expr_data.x_val.item()
        y = expr_data.y_val.item()
        z = expr_data.z_val.item()

        print("\n<<< Data >>>")
        print("x:", x)
        print("y:", y)
        print("z:", z)
        print(f"True expression: {expr_data.expr_str} = {eval(expr_data.expr_str)}")

        ### brute force
        print("\n<<< Brute Force >>>")
        solved, bf_steps, duration, expr = brute_force_exp(x, y, z, expr_data)

        if solved:
            print(f"Found expression: {expr} = {z}")
        else:
            print("No expression found")
        print(f"Computation time: {duration:0.5f} Seconds")
        print(f"Brute-force steps: {bf_steps}")


        # gnn brute force with extra steps
        print("\n<<< GNN Brute Force >>>")
        model = GATNetwork(expr_data.num_features, 16, expr_data.num_classes).to(device)
        if os.path.isfile(MODEL_PATH):
            model_state = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
            model.load_state_dict(model_state["model"], strict=True)

        gnn_solved, gnn_bf_steps, gnn_duration, gnn_expr = gnn_brute_force_exp(model, expr_data)

        if gnn_solved:
            print(f"Found expression: {gnn_expr} = {z}")
        else:
            print("No expression found")
        print(f"Computation time: {gnn_duration:0.5f} Seconds")
        print(f"Brute-force steps: {gnn_bf_steps}")

        if gnn_bf_steps < bf_steps:
            gnn_faster += 1
            avg_percent += (bf_steps / gnn_bf_steps)
            print(f"\n==> GNN Brute Force is faster by {(bf_steps / gnn_bf_steps):0.1f} %")
        else:
            print(f"\n==> Brute Force is faster by {(gnn_bf_steps / bf_steps):0.1f} %")

    print(f"\nTotal times GNN was faster: {gnn_faster}/100")
    print(f"Average percentage of GNN being faster: {(avg_percent / gnn_faster):0.1f} %")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="GPU", type=int, default=-1)
    args = parser.parse_args()
    main(**vars(args))


