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
    # 8: "NOP",  # identity mapping / no operation
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


def gen_big_expr_data(testing: bool) -> Iterator[Data]:
    edge_index = torch.tensor(
        # 1 is the operation and 3 the result node
        [[0, 1],
         [2, 1],
         [1, 3],
         [3, 0],
         [3, 2],
        # 5 is the operation  and 7 the result node
         [4, 5],
         [6, 5],
         [5, 7],
         [7, 4],
         [7, 6],
        # 9 is the operation and 11 the result node
         [8, 9],
         [10, 9],
         [9, 11],
         [11, 8],
         [11, 10],
        # 13 is the operation and 15 the result node
         [12, 13],
         [14, 13],
         [13, 15],
         [15, 12],
         [15, 14],
        # 16 is the operation and 17 is the result node
         [3, 16],
         [7, 16],
         [16, 17],
         [17, 3],
         [17, 7],
        # 18 is the operation and 19 the result node
         [11, 18],
         [15, 18],
         [18, 19],
         [19, 11],
         [19, 15],
        # 20 is the operation and 21 the result node
         [17, 20],
         [19, 20],
         [20, 21],
         [21, 17],
         [21, 19]],
        dtype=torch.long).t().contiguous()

    while True:
        train_mask = torch.tensor([False, True, False, False,
                                   False, True, False, False,
                                   False, True, False, False,
                                   False, True, False, False,
                                   True, False, True, False,
                                   True, False], dtype=torch.bool)

        test_mask = torch.tensor([False, True, False, False,
                                  False, True, False, False,
                                  False, True, False, False,
                                  False, True, False, False,
                                  True, False, True, False,
                                  True, False], dtype=torch.bool)

        x_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)
        y_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)

        op_0 = torch.randint(0, len(operation_dict), (1,)).item()
        op_1 = torch.randint(0, len(operation_dict), (1,)).item()
        op_2 = torch.randint(0, len(operation_dict), (1,)).item()
        op_3 = torch.randint(0, len(operation_dict), (1,)).item()
        op_4 = torch.randint(0, len(operation_dict), (1,)).item()
        op_5 = torch.randint(0, len(operation_dict), (1,)).item()
        op_6 = torch.randint(0, len(operation_dict), (1,)).item()

        match op_0:
            case 0: z_val_0 = x_val + y_val
            case 1: z_val_0 = x_val - y_val
            case 2: z_val_0 = x_val * y_val
            case 3: z_val_0 = x_val & y_val
            case 4: z_val_0 = x_val | y_val
            case 5: z_val_0 = x_val ^ y_val
            case 6: z_val_0 = x_val >> y_val
            case 7: z_val_0 = x_val << y_val
        
        match op_1:
            case 0: z_val_1 = x_val + y_val
            case 1: z_val_1 = x_val - y_val
            case 2: z_val_1 = x_val * y_val
            case 3: z_val_1 = x_val & y_val
            case 4: z_val_1 = x_val | y_val
            case 5: z_val_1 = x_val ^ y_val
            case 6: z_val_1 = x_val >> y_val
            case 7: z_val_1 = x_val << y_val

        match op_2:
            case 0: z_val_2 = x_val + y_val
            case 1: z_val_2 = x_val - y_val
            case 2: z_val_2 = x_val * y_val
            case 3: z_val_2 = x_val & y_val
            case 4: z_val_2 = x_val | y_val
            case 5: z_val_2 = x_val ^ y_val
            case 6: z_val_2 = x_val >> y_val
            case 7: z_val_2 = x_val << y_val

        match op_3:
            case 0: z_val_3 = x_val + y_val
            case 1: z_val_3 = x_val - y_val
            case 2: z_val_3 = x_val * y_val
            case 3: z_val_3 = x_val & y_val
            case 4: z_val_3 = x_val | y_val
            case 5: z_val_3 = x_val ^ y_val
            case 6: z_val_3 = x_val >> y_val
            case 7: z_val_3 = x_val << y_val

        match op_4:
            case 0: z_val_4 = z_val_0 + z_val_1
            case 1: z_val_4 = z_val_0 - z_val_1
            case 2: z_val_4 = z_val_0 * z_val_1
            case 3: z_val_4 = z_val_0 & z_val_1
            case 4: z_val_4 = z_val_0 | z_val_1
            case 5: z_val_4 = z_val_0 ^ z_val_1
            case 6: z_val_4 = z_val_0 >> z_val_1
            case 7: z_val_4 = z_val_0 << z_val_1

        match op_5:
            case 0: z_val_5 = z_val_2 + z_val_3
            case 1: z_val_5 = z_val_2 - z_val_3
            case 2: z_val_5 = z_val_2 * z_val_3
            case 3: z_val_5 = z_val_2 & z_val_3
            case 4: z_val_5 = z_val_2 | z_val_3
            case 5: z_val_5 = z_val_2 ^ z_val_3
            case 6: z_val_5 = z_val_2 >> z_val_3
            case 7: z_val_5 = z_val_2 << z_val_3

        match op_6:
            case 0: z_val = z_val_4 + z_val_5
            case 1: z_val = z_val_4 - z_val_5
            case 2: z_val = z_val_4 * z_val_5
            case 3: z_val = z_val_4 & z_val_5
            case 4: z_val = z_val_4 | z_val_5
            case 5: z_val = z_val_4 ^ z_val_5
            case 6: z_val = z_val_4 >> z_val_5
            case 7: z_val = z_val_4 << z_val_5

        # if testing is enabled, the interim results will not be provided
        if testing:
            x = torch.tensor([[x_val], [-1.], [y_val], [0],
                              [x_val], [-1.], [y_val], [0],
                              [x_val], [-1.], [y_val], [0],
                              [x_val], [-1.], [y_val], [0],
                              [-1.], [z_val_4], [-1.], [0],
                              [-1.], [z_val]], dtype=torch.float)
        else:
            x = torch.tensor([[x_val], [-1.], [y_val], [z_val_0],
                            [x_val], [-1.], [y_val], [z_val_1],
                            [x_val], [-1.], [y_val], [z_val_2],
                            [x_val], [-1.], [y_val], [z_val_3],
                            [-1.], [z_val_4], [-1.], [z_val_5],
                            [-1.], [z_val]], dtype=torch.float)

        y = torch.tensor([op_0, op_1, op_2, op_3, op_4, op_5, op_6], dtype=torch.long)

        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.num_classes = len(operation_dict)

        yield data
