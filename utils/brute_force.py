"""library module for providing brute force functions"""
import time
import re
from itertools import product
from copy import copy
from typing import Tuple
import torch
from torch import nn
from torch_geometric.data import Data

from .datasets import expr_dict

operation_dict = {
    0: "add",
    1: "sub",
    2: "mul",
    3: "and",
    4: "or",
    5: "xor"
}


def brute_force_exp(x: int, y: int, z: int, orig_expr_str: str,
                    placeholder_str: str, num_ops: int) -> Tuple[bool, int, float, str]:
    """
    Brute force function which iterates over every possible operation and evaluates it.
    Arguments:
        x: X input value
        y: Y input value
        z: Z output value
        expr_str: expression string from the graph
        placeholder_str: expression string with placeholder operations
        num_ops: number of operations
    Returns:
        Tuple[bool, int, float, str]: solved status, brute force steps and brute force computation time and expr.
    """
    valid_num_ops = [2**val-1 for val in range(0, num_ops)]
    assert num_ops in valid_num_ops, "Number of operations must be valid by the term 2^n-1"
    
    steps = 0
    duration = 0.

    # iterate over every combination of operations for the whole expression
    start = time.perf_counter()
    for op_comb in product(operation_dict.keys(), repeat=num_ops):
        steps += 1
        brute_expr_str = copy(placeholder_str)
        # replace the placeholder operations accordingly
        for idx, op in enumerate(op_comb):
            brute_expr_str = brute_expr_str.replace(f"op{idx}", f"{expr_dict[op]}", 1)
        
        try:
            brute_expr_str_w_vals = copy(brute_expr_str)
            brute_expr_str_w_vals = brute_expr_str_w_vals.replace("x", f"{x}")
            brute_expr_str_w_vals = brute_expr_str_w_vals.replace("y", f"{y}")
            result = eval(brute_expr_str_w_vals)
        except Exception as _:
            result = None

        if result == z:
            # evaluate with other values
            if re_eval(brute_expr_str, orig_expr_str):
                end = time.perf_counter()
                duration = end - start
                return True, steps, duration, brute_expr_str
            else: 
                continue
        else:
            continue

    end = time.perf_counter()
    duration = end - start
    return False, steps, duration, brute_expr_str


def gnn_brute_force_exp(gnn: nn.Sequential, data: Data, num_ops: int) -> Tuple[bool, int, float, str]:
    """
    Brute force function which utilizes the GNN predicted probabilities for the sinle operations.
    Arguments:
        gnn: the gnn to predict the operations
        data: graph input data
        num_ops: number of operations
    Returns:
        Tuple[bool, int, float, str]: solved status, brute force steps and brute force computation time and expr.
    """
    valid_num_ops = [2**val-1 for val in range(0, num_ops)]
    assert num_ops in valid_num_ops, "Number of operations must be valid by the term 2^n-1"
    
    x = data.x_val.item()
    y = data.y_val.item()
    z = data.z_val
    orig_expr_str = data.expr_str

    steps = 0
    duration = 0.

    with torch.no_grad():
        prediction = gnn(data, data.edge_index).squeeze()

    ops = []
    topk_ops = torch.topk(prediction, len(expr_dict))
    for i in range(len(data.operations)):
        tmp_op = [(item0.item(), item1.item()) for item0, item1 in zip(topk_ops.indices[i],
                	                                                   torch.exp(topk_ops.values[i]))]
        tmp_op.sort(key=lambda tuple: tuple[1], reverse=True)
        ops.append(tmp_op)

    # iterate over every combination of operations for the whole expression
    start = time.perf_counter()
    for op_comb in product(operation_dict.keys(), repeat=num_ops):
        steps += 1
        brute_expr_str = copy(data.expr_str_placeholder)
        # replace the placeholder operations accordingly
        for idx, op in enumerate(op_comb):
            brute_expr_str = re.sub(r"\b%s\b" % f"op{idx}",
                                    f"{expr_dict[op]}",
                                    brute_expr_str, 1)
            # brute_expr_str = brute_expr_str.replace(f"op{idx}", f"{expr_dict[op]}", 1)

        try:
            brute_expr_str_w_vals = copy(brute_expr_str)
            brute_expr_str_w_vals = brute_expr_str_w_vals.replace("x", f"{x}")
            brute_expr_str_w_vals = brute_expr_str_w_vals.replace("y", f"{y}")
            result = eval(brute_expr_str_w_vals)
        except Exception as _:
            result = None

        if result == z:
            # evaluate with other values
            if re_eval(brute_expr_str, orig_expr_str):
                end = time.perf_counter()
                duration = end - start
                return True, steps, duration, brute_expr_str
            else:
                continue
        else:
            continue

    end = time.perf_counter()
    duration = end - start
    return False, steps, duration, brute_expr_str


def re_eval(expr: str, expr_orig: str) -> bool:
    """
    Helper function which re-evaluates the expression with other values to ensure its correctness. 
    """
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
        # print(f"{expr_a} = {z_val} || {expr_b} = {true_z_val}")
        if z_val != true_z_val:
            return False

    return True

