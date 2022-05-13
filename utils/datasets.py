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
    operations_list = [0, 1, 2]

    while True:
        edge_index = torch.tensor([[0, 1, 2], [1, 3, 1]], dtype=torch.long)

        x_val = torch.randint(-10, 10, (1,))
        y_val = torch.randint(-10, 10, (1,))

        chosen_operation = operations_list[torch.randint(0, len(operations_list), (1,))]
        match chosen_operation:
            case 0: z_val = x_val + y_val
            case 1: z_val = x_val - y_val
            case 2: z_val = x_val * y_val
            #case 3: z_val = x_val / y_val

        x = torch.tensor([[x_val], [chosen_operation], [y_val], [z_val]], dtype=torch.float)
        y = torch.tensor([[chosen_operation]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)

        data.train_mask = torch.tensor([False, True, False, False], dtype=torch.bool)
        data.test_mask = torch.tensor([False, True, False, False], dtype=torch.bool)
        data.num_classes = torch.tensor([len(operations_list)], dtype=torch.long)
        
        yield data