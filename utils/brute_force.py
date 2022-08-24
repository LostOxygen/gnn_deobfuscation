"""library module for providing brute force functions"""
from re import X
import time
from typing import Tuple
import torch
from torch import nn
from torch_geometric.data import Data

from utils.datasets import operation_dict

operation_dict = {
    0: "+",
    1: "-",
    2: "*",
    3: "&",
    4: "|",
    5: "^",
    #6: ">>"
}


def brute_force_exp(x: int, y: int, z: int) -> Tuple[bool, int, float, str]:
    """
    Arguments:
        x: X input value
        y: Y input value
        z: Z input value
    Returns:
        Tuple[bool, int, float, str]: solved status, brute force steps and brute force computation time and expr.
    """
    steps = 0
    duration = 0.

    start = time.perf_counter()
    for op0 in operation_dict.values():
        for op1 in operation_dict.values():
            for op2 in operation_dict.values():
                for op3 in operation_dict.values():
                    for op4 in operation_dict.values():
                        for op5 in operation_dict.values():
                            for op6 in operation_dict.values():
                                steps += 1
                                expr = f"(({x} {op0} {y}) {op4} ({x} {op1} {y})) {op6} " \
                                       f"(({x} {op2} {y}) {op5} ({x} {op3} {y}))"
                                try:
                                    result = eval(expr)
                                except Exception as error:
                                    #print(error)
                                    result = None
                                if result == z:
                                    end = time.perf_counter()
                                    duration = end - start
                                    return True, steps, duration, expr
                                else:
                                    continue
    end = time.perf_counter()
    duration = end - start
    return False, steps, duration, expr


def gnn_brute_force_exp(gnn: nn.Sequential, data: Data) -> Tuple[bool, int, float, str]:
    """
    Arguments:
        data: graph input data
    Returns:
        Tuple[bool, int, float, str]: solved status, brute force steps and brute force computation time and expr.
    """
    x = data.x_val.item()
    y = data.y_val.item()
    z = data.z_val.item()

    steps = 0
    duration = 0.

    with torch.no_grad():
        prediction = gnn(data.x, data.edge_index)

    ops = []
    topk_ops = torch.topk(prediction, len(operation_dict))
    for i in range(7):
        tmp_op = [(item0.item(), item1.item()) for item0, item1 in zip(topk_ops.indices[i],
                	                                                   torch.exp(topk_ops.values[i]))]
        tmp_op.sort(key=lambda tuple: tuple[1], reverse=True)
        ops.append(tmp_op)

    start = time.perf_counter()
    for op0 in ops[0]:
        for op1 in ops[1]:
            for op2 in ops[2]:
                for op3 in ops[3]:
                    for op4 in ops[4]:
                        for op5 in ops[5]:
                            for op6 in ops[6]:
                                steps += 1
                                expr = f"(({x} {operation_dict[op0[0]]} {y}) {operation_dict[op4[0]]} " \
                                       f"({x} {operation_dict[op1[0]]} {y})) {operation_dict[op6[0]]} " \
                                       f"(({x} {operation_dict[op2[0]]} {y}) {operation_dict[op5[0]]} " \
                                       f"({x} {operation_dict[op3[0]]} {y}))"
                                try:
                                    result = eval(expr)
                                except Exception as error:
                                    #print(error)
                                    result = None
                                if result == z:
                                    end = time.perf_counter()
                                    duration = end - start
                                    return True, steps, duration, expr
                                else:
                                    continue
    end = time.perf_counter()
    duration = end - start
    return False, steps, duration, expr
