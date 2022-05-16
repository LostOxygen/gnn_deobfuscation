"""library module for providing datasets"""
import torch
from torch_geometric.data import Data


def gen_expr_data(batch_size: int) -> Data:
    """
    Helper function to yield a data object for a simple expression like x+y=z

    Paramters:
        batch_size: the batch size of the data object
    
    Returns:
        Data object for the given dataset
    """
    operations_list = [0, 1, 2]
    edge_index = torch.tensor([[0, 1], [1, 3], [2, 1]], dtype=torch.long).t().contiguous()

    while True:
        x_list = []
        y_list = []
        train_mask_list = [torch.tensor([False, True, False, False], dtype=torch.bool)] * batch_size
        test_mask_list = [torch.tensor([False, True, False, False], dtype=torch.bool)] * batch_size


        for _ in range(batch_size):
            x_val = torch.randint(-10, 10, (1,))
            y_val = torch.randint(-10, 10, (1,))

            chosen_operation = operations_list[torch.randint(0, len(operations_list), (1,))]
            match chosen_operation:
                case 0: z_val = x_val + y_val
                case 1: z_val = x_val - y_val
                case 2: z_val = x_val * y_val
                #case 3: z_val = x_val / y_val

            x_list.append(torch.tensor([[x_val], [chosen_operation], [y_val], [z_val]], dtype=torch.float))
            y_list.append(torch.tensor([x_val, chosen_operation, y_val, z_val], dtype=torch.long))
           
           
        data = Data(x=torch.cat(x_list), y=torch.cat(y_list), edge_index=edge_index)
        data.train_mask = torch.cat(train_mask_list)
        data.test_mask = torch.cat(test_mask_list)
        data.num_classes = len(operations_list)

        yield data