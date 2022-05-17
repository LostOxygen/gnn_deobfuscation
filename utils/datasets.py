"""library module for providing datasets"""
import torch
from torch_geometric.data import Data


def gen_expr_data() -> Data:
    """
    Helper function to yield a data object for a simple expression like x+y=z

    Paramters:
        None
    
    Returns:
        Data object for the given dataset
    """
    operations_list = [0, 1, 2]  # for 0: add, 1: sub, 2: mul, -1: no operation
    edge_index = torch.tensor([
            [0, 1],
            [1, 3],
            [2, 1]], dtype=torch.long
        ).t().contiguous()

    while True:
        train_mask = torch.tensor([True, False, True, True], dtype=torch.bool)
        test_mask = torch.tensor([False, True, False, False], dtype=torch.bool)


        x_val = torch.randint(-100, 100, (1,))
        y_val = torch.randint(-100, 100, (1,))

        chosen_operation = operations_list[torch.randint(0, len(operations_list), (1,))]
        match chosen_operation:
            case 0: z_val = x_val + y_val
            case 1: z_val = x_val - y_val
            case 2: z_val = x_val * y_val
            #case 3: z_val = x_val / y_val

        x = torch.tensor([[x_val], [0], [y_val], [z_val]], dtype=torch.float)
        y = torch.tensor([chosen_operation], dtype=torch.long)


        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.num_classes = len(operations_list)

        yield data