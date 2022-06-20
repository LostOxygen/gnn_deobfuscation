"""library module for providing datasets"""
from typing import Iterator
import math
import torch
from torch_geometric.data import Data

operation_dict = {
    0: "add",
    1: "sub",
    2: "mul",
    3: "div",
    4: "and",
    5: "or",
    6: "xor",
    7: "shl",
    8: "shr",
    # 9: "not",
}

class IOSamplesDataset(torch.utils.data.Dataset):
    """
    Class for I/O Samples packed in a pytorch Dataset.
    """
    def __init__(self):
        self.edge_index = torch.tensor(
                               [[0, 1],
                                [1, 3],
                                [2, 1]],
                                dtype=torch.long).t().contiguous()
    
    def __getitem__(self, idx):
        x_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)
        y_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)

        chosen_operation = torch.randint(0, len(operation_dict), (1,)).item()
        match chosen_operation:
            case 0: z_val = x_val + y_val
            case 1: z_val = x_val - y_val
            case 2: z_val = x_val * y_val
            case 3:
                if y_val == 0:
                    y_val += 1
                z_val = x_val / y_val
            case 4: z_val = x_val & y_val
            case 5: z_val = x_val | y_val
            case 6: z_val = x_val ^ y_val
            case 7: z_val = x_val << y_val
            case 8: z_val = x_val >> y_val
            case 9:
                z_val = ~x_val
                y_val = 0

        x = torch.tensor([[x_val], [0.], [y_val], [z_val]], dtype=torch.float)
        y = torch.tensor(chosen_operation, dtype=torch.long)

        return x, y, self.edge_index

    def __len__(self):
        return int(1e9)


def gen_expr_data() -> Iterator[Data]:
    """
    Helper function to yield a data object for a simple expression like x+y=z
    0: addition
    1: subtraction
    2: multiplication
    3: division
    4: logical and
    5: logical or
    6: logical xor
    7: logical shift left
    8: logical shift right
    9: logical not

    Paramters:
        None
    
    Returns:
        Data object for the given dataset
    """

    edge_index = torch.tensor([
        [0, 1],
        [1, 3],
        [2, 1]], dtype=torch.long
    ).t().contiguous()

    while True:
        train_mask = torch.tensor(
            [False, True, False, False], dtype=torch.bool)
        test_mask = torch.tensor([False, True, False, False], dtype=torch.bool)

        x_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)
        y_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)

        chosen_operation = torch.randint(0, len(operation_dict), (1,)).item()

        match chosen_operation:
            case 0: z_val = x_val + y_val
            case 1: z_val = x_val - y_val
            case 2: z_val = x_val * y_val
            case 3:
                if y_val == 0:
                    y_val += 1
                z_val = x_val / y_val
            case 4: z_val = x_val & y_val
            case 5: z_val = x_val | y_val
            case 6: z_val = x_val ^ y_val
            case 7: z_val = x_val << y_val
            case 8: z_val = x_val >> y_val
            case 9:
                z_val = ~x_val
                y_val = 0

        x = torch.tensor([[x_val], [0.], [y_val], [z_val]], dtype=torch.float)
        y = torch.tensor([chosen_operation], dtype=torch.long)

        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.num_classes = len(operation_dict)

        yield data
