"""library module for providing datasets"""
from typing import Iterator
import torch
from torch_geometric.data import Data

operation_dict = {
    0: "add",
    1: "sub",
    2: "mul",
    3: "and",
    4: "or",
    5: "xor",
    6: "shr",
    # 7: "shl",
    # 8: "NOP",  # this is no operation
}

def gen_expr_data() -> Iterator[Data]:
    edge_index = torch.tensor(
        [[0, 1],
         [2, 1],
         [1, 3],
         [3, 0],
         [3, 2]],
        dtype=torch.long).t().contiguous()

    while True:
        train_mask = torch.tensor([False, True, False, False], dtype=torch.bool)
        test_mask = torch.tensor([False, True, False, False], dtype=torch.bool)

        x_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)
        y_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)

        operation = torch.randint(0, len(operation_dict), (1,)).item()

        match operation:
            case 0: z_val = x_val + y_val
            case 1: z_val = x_val - y_val
            case 2: z_val = x_val * y_val
            case 3: z_val = x_val & y_val
            case 4: z_val = x_val | y_val
            case 5: z_val = x_val ^ y_val
            case 6: z_val = x_val >> y_val
            case 7: z_val = x_val << y_val

            
        x = torch.tensor([[x_val], [-1.], [y_val], [z_val]], dtype=torch.float)
        y = torch.tensor([operation], dtype=torch.long)

        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.num_classes = len(operation_dict)

        yield data
