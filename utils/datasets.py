"""library module for providing datasets"""
from typing import Iterator
import math
import torch
from torch_geometric.data import Data

operation_dict = {
    0: "add",
    1: "sub",
    2: "mul",
    3: "and",
    4: "or",
    5: "xor",
    # 6: "NOP",  # this is no operation
    # 7: "div",
    # 8: "shl",
    # 9: "shr",

}

class IOSamplesDataset(torch.utils.data.Dataset):
    """
    Class for I/O Samples packed in a pytorch Dataset.
    """
    def __init__(self):
        self.edge_index = torch.tensor(
                               [[0, 2],
                                [0, 3],
                                [1, 2],
                                [1, 3],
                                [2, 4],
                                [3, 4],
                                [4, 5]],
                                dtype=torch.long).t().contiguous()
        self.num_classes = len(operation_dict)
        self.num_features = 1
        self.train_mask = torch.tensor([False, False, True, True, True, False], dtype=torch.bool)
        self.test_mask = torch.tensor([False, False, True, True, True, False], dtype=torch.bool)
    
    def __getitem__(self, idx): # pylint: ignore=unused-argument
        x_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)
        y_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)

        chosen_operation1 = torch.randint(0, self.num_classes-1, (1,)).item()
        chosen_operation2 = torch.randint(0, self.num_classes-1, (1,)).item()
        chosen_operation3 = torch.randint(0, self.num_classes-1, (1,)).item()

        z_tmp1 = self._get_expression_result(chosen_operation1, x_val, y_val).to(torch.int32)
        z_tmp2 = self._get_expression_result(chosen_operation2, x_val, y_val).to(torch.int32)
        z_val = self._get_expression_result(chosen_operation3, z_tmp1, z_tmp2)

        x = torch.tensor([[x_val], [y_val], [0.], [0.], [0.], [z_val]], dtype=torch.float)
        y = torch.tensor([9, 9, chosen_operation1, chosen_operation2, chosen_operation3, 9], dtype=torch.long)

        return x, y, self.edge_index

    def _get_expression_result(self, op: int, val1: int, val2: int) -> int:
        match op:
            case 0: z_val = val1 + val2
            case 1: z_val = val1 - val2
            case 2: z_val = val1 * val2
            case 3:
                if val2 == 0:
                    val2 += 1
                z_val = val1 / val2
            case 4: z_val = val1 & val2
            case 5: z_val = val1 | val2
            case 6: z_val = val1 ^ val2
            case 7: z_val = val1 << val2
            case 8: z_val = val1 >> val2
        
        return z_val

    def __len__(self):
        return int(1e9)

def gen_expr_data(operation: int=None) -> Iterator[Data]:
    while True:
        edge_index = torch.tensor(
                [[0, 1],
                 [2, 1],
                 [1, 3],
                 [3, 0],
                 [3, 2]],
            dtype=torch.long).t().contiguous()

        train_mask = torch.tensor([False, True, False, False], dtype=torch.bool)
        test_mask = torch.tensor([False, True, False, False], dtype=torch.bool)

        x_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)
        y_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)

        if operation is None:
            operation = torch.randint(0, len(operation_dict), (1,)).item()

        match operation:
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

        x = torch.tensor([[x_val], [-1.], [y_val], [z_val]], dtype=torch.float)
        y = torch.tensor([operation], dtype=torch.long)

        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.num_classes = len(operation_dict)

        yield data
