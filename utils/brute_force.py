"""library module for providing brute force functions"""
import time
from copy import copy
from typing import Tuple
import torch
from torch import nn
from torch_geometric.data import Data

from utils.datasets import expr_dict



def brute_force_exp(x: int, y: int, z: int, data: Data) -> Tuple[bool, int, float, str]:
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
    expr_str_orig = data.expr_str_orig

    start = time.perf_counter()
    for op0 in expr_dict.values():
        for op1 in expr_dict.values():
            for op2 in expr_dict.values():
                for op3 in expr_dict.values():
                    for op4 in expr_dict.values():
                        for op5 in expr_dict.values():
                            for op6 in expr_dict.values():
                                steps += 1
                                expr = f"(({x} {op0} {y}) {op4} ({x} {op1} {y})) {op6} " \
                                       f"(({x} {op2} {y}) {op5} ({x} {op3} {y}))"
                                expr_clean = f"((x {op0} y) {op4} (x {op1} y)) {op6} " \
                                             f"((x {op2} y) {op5} (x {op3} y))"
                                try:
                                    result = eval(expr)
                                except Exception as _:
                                    result = None
                                if result == z:
                                    # evaluate with other values
                                    if re_eval(expr_clean, expr_str_orig):
                                        end = time.perf_counter()
                                        duration = end - start
                                        return True, steps, duration, expr
                                    else:
                                        continue
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
    expr_str_orig = data.expr_str_orig

    steps = 0
    duration = 0.

    with torch.no_grad():
        prediction = gnn(data.x, data.edge_index)

    ops = []
    topk_ops = torch.topk(prediction, len(expr_dict))
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
                                expr = f"(({x} {expr_dict[op0[0]]} {y}) {expr_dict[op4[0]]} " \
                                       f"({x} {expr_dict[op1[0]]} {y})) {expr_dict[op6[0]]} " \
                                       f"(({x} {expr_dict[op2[0]]} {y}) {expr_dict[op5[0]]} " \
                                       f"({x} {expr_dict[op3[0]]} {y}))"
                                expr_clean = f"((x {expr_dict[op0[0]]} y) {expr_dict[op4[0]]} " \
                                             f"(x {expr_dict[op1[0]]} y)) {expr_dict[op6[0]]} " \
                                             f"((x {expr_dict[op2[0]]} y) {expr_dict[op5[0]]} " \
                                             f"(x {expr_dict[op3[0]]} y))"
                                try:
                                    result = eval(expr)
                                except Exception as _:
                                    result = None
                                if result == z:
                                    # evaluate with other values
                                    if re_eval(expr_clean, expr_str_orig):
                                        end = time.perf_counter()
                                        duration = end - start
                                        return True, steps, duration, expr
                                    else: 
                                        continue
                                else: 
                                    continue
    end = time.perf_counter()
    duration = end - start
    return False, steps, duration, expr


def re_eval(expr: str, expr_orig: str) -> bool:
    for _ in range(10000):
        expr_a = copy(expr)
        expr_b = copy(expr_orig)
        x_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int).item()
        y_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int).item()
        expr_a = expr_a.replace("x", str(x_val))
        expr_a = expr_a.replace("y", str(y_val))
        expr_b = expr_b.replace("x", str(x_val))
        expr_b = expr_b.replace("y", str(y_val))

        z_val = eval(expr_a)
        true_z_val = eval(expr_b)
        #print(f"{expr_a} = {z_val} || {expr_b} = {true_z_val}")
        if z_val != true_z_val:
            return False

    return True

