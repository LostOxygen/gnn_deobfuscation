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
    #6: "NOP"  # identity mapping / no operation
}

expr_dict = {
    0: "+",
    1: "-",
    2: "*",
    3: "&",
    4: "|",
    5: "^",
    #6: " "
}

def gen_expr_graph(num_operations: int) -> Iterator[Data]:
    """
    Returns a generator for graph data with num_operations operators.
    """
    valid_num_ops = [2**val-1 for val in range(0, num_operations)]
    assert num_operations in valid_num_ops, "Number of operations must be valid by the term 2^n-1"
    
    num_input_vars = num_operations + 1
    assert num_input_vars % 2 == 0, "Number of input variables must be even"
    num_nodes = num_input_vars + (2*num_operations)
    nodes_per_stage = [int(num_input_vars)]
    input_nodes = []
    output_nodes = [int(num_nodes - 1)]
    operation_nodes = []
    interim_nodes = []

    # create the number of nodes per stage in the graph
    node_counter = num_input_vars / 2
    while node_counter >= 1:
        nodes_per_stage.append(int(node_counter))
        nodes_per_stage.append(int(node_counter))
        node_counter /= 2
    
    # check how many stages of the graph contain operation nodes
    num_stages_with_ops = 0
    ops_counter = num_operations
    while ops_counter >= 1:
        num_stages_with_ops += 1
        ops_counter /= 2

    # add the nodes to their respective list
    node_switch = "input" # switch variable to switch between node types
    current_node = 0
    current_tree_stage = 0
    curr_len_ops = 0
    curr_len_inter = 0

    while current_node <= num_nodes-2:
        #print(f"stage: {current_tree_stage}, node: {current_node}, switch: {node_switch}, len ops: {curr_len_ops}, len inter: {curr_len_inter}")
        match node_switch:
            case "input": 
                input_nodes.append(current_node)
                if len(input_nodes) >= num_input_vars:
                    node_switch = "operation"
                    current_tree_stage += 1
            
            case "operation": 
                operation_nodes.append(current_node)
                if len(operation_nodes) >= curr_len_ops+nodes_per_stage[current_tree_stage]:
                    node_switch = "interim"           
                    curr_len_ops = len(operation_nodes)
                    current_tree_stage += 1
            
            case "interim": 
                interim_nodes.append(current_node)
                if len(interim_nodes) >= curr_len_inter+nodes_per_stage[current_tree_stage]:
                    node_switch = "operation"
                    curr_len_inter = len(interim_nodes)
                    current_tree_stage += 1
        
        current_node += 1        
        
    # print("num input variables: ", num_input_vars)
    # print("num total nodes: ", num_nodes)
    # print("nodes per stage: ", nodes_per_stage)
    # print("input nodes: ", input_nodes)
    # print("output nodes: ", output_nodes)
    # print("operation nodes: ", operation_nodes)
    # print("interim nodes: ", interim_nodes)

    edge_index = []
    # add the input nodes and their edge into the edge list
    for idx, ido in zip(range(0, len(input_nodes), 2), \
        range(0, len(operation_nodes))):
        edge_index.append([input_nodes[idx], operation_nodes[ido]])
        edge_index.append([input_nodes[idx+1], operation_nodes[ido]])

    # add the operation nodes and their edges into the edge list without the result node
    for idx in range(0, len(operation_nodes)-1):
        edge_index.append([operation_nodes[idx], interim_nodes[idx]])

    # add the interim nodes and their edges into the edge list
    # but skip the first stage of operation nodes in the graph
    for idx, op_node in zip(range(0, len(interim_nodes), 2), \
                            operation_nodes[nodes_per_stage[1]:]):
        edge_index.append([interim_nodes[idx], op_node])
        edge_index.append([interim_nodes[idx+1], op_node])

    # finally add the last operation to the output node edge
    edge_index.append([operation_nodes[-1], output_nodes[0]])
    # print(f"edge list: {edge_index}")
    # convert edge indizes to a torch tensor and transpose them
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # create the masks for the train/test    
    mask_values = [False if idx not in operation_nodes else True for idx in range(0, num_nodes)]
    mask = torch.tensor(mask_values, dtype=torch.bool)

    # build the expression string
    expr_str = ""
    op_insert_state = "left" # state machine to keep track of the next ops position
    str_state = "left"
    left_str = ""
    right_str = ""
    for input_idx in range(0, num_operations):
        match op_insert_state:
            case "left": 
                if str_state == "left":
                    left_str += f"((x op{input_idx} y)"
                elif str_state == "right":
                    right_str += f"((x op{input_idx} y)"
                op_insert_state = "right"

            case "middle":
                if str_state == "left":
                    left_str = left_str[:int(len(left_str)/2)] + f" op{input_idx} " + left_str[int(len(left_str)/2):]
                elif str_state == "right":
                    right_str = right_str[:int(
                        len(right_str)/2)] + f" op{input_idx} " + right_str[int(len(right_str)/2):]
                
                if len(left_str) == len(right_str):
                    expr_str = "(" + left_str + f" op{input_idx+1} " + right_str + ")"
                    left_str = ""
                    right_str = ""
                elif len(left_str):
                    str_state = "right"

                op_insert_state = "left"

            case "right":
                if str_state == "left":
                    left_str += f"(x op{input_idx} y))"
                elif str_state == "right":
                    right_str += f"(x op{input_idx} y))"
                op_insert_state = "middle"
        
    if len(left_str) and len(right_str) == 0:
        expr_str = left_str
    
    # print(f"expression string: {expr_str}")

    # generate the actual data
    tmp_vals = [[-1]]*num_nodes
    while True:
        # get input variables
        x_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)
        y_val = torch.randint(0, 2**8-1, (1,), dtype=torch.int)

        # get randomized operations
        operations = torch.zeros(num_operations, dtype=torch.long)
        interim_results = torch.zeros(len(interim_nodes), dtype=torch.int32)
        # fill the operation vector accordingly
        for i in range(num_operations):
            operations[i] = torch.randint(0, len(operation_dict), (1,)).item()

        # fill the first stage of interim results according to the input vars and their operations
        for idx in range(nodes_per_stage[1]):
            interim_results[idx] = match_op(operations[idx], x_val, y_val)
        
        for idx, idi in zip(range(nodes_per_stage[1], len(interim_nodes)), \
                            range(0, len(interim_nodes), 2)):
            interim_results[idx] = match_op(operations[idx],
                                            interim_nodes[idi],
                                            interim_nodes[idi+1])

        z_val = match_op(interim_nodes[-2], interim_nodes[-1], operations[-1])
        
        # print(f"operation vector: {[operation_dict[i.item()] for i in operations]}")
        # print(f"x_val: {x_val} \ny_val: {y_val}")
        # print(f"interim vector: {interim_results}")

        # create the data tensors and write the node values into the data object
        x = torch.tensor(tmp_vals, dtype=torch.float)
        # input variables
        x[torch.arange(0, num_input_vars, 2)][0] = x_val.float()
        x[torch.arange(1, num_input_vars, 2)][0] = y_val.float()
        # output variables
        x[-1][0] = z_val
        y = operations
        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = mask
        data.test_mask = mask
        data.num_classes = len(operation_dict)
        data.z_val = z_val
        data.expr_str = expr_str
        data.operations = operations

        yield data


def gen_big_expr_data(testing: bool) -> Iterator[Data]:
    # create the graph as an edge list tensor
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
         [21, 19],
        # additional inverse edges
         [21, 0],
         [21, 2],
         [21, 4],
         [21, 6],
         [21, 8],
         [21, 10],
         [21, 12],
         [21, 14]],
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

        # choose operations
        op_0 = torch.randint(0, len(operation_dict), (1,)).item()
        op_1 = torch.randint(0, len(operation_dict), (1,)).item()
        op_2 = torch.randint(0, len(operation_dict), (1,)).item()
        op_3 = torch.randint(0, len(operation_dict), (1,)).item()
        op_4 = torch.randint(0, len(operation_dict), (1,)).item()
        op_5 = torch.randint(0, len(operation_dict), (1,)).item()
        op_6 = torch.randint(0, len(operation_dict), (1,)).item()

        # calculate the interim and final results 
        z_val_0 = match_op(op_0, x_val, y_val)
        z_val_1 = match_op(op_1, x_val, y_val)
        z_val_2 = match_op(op_2, x_val, y_val)
        z_val_3 = match_op(op_3, x_val, y_val)
        z_val_4 = match_op(op_4, z_val_0, z_val_1)
        z_val_5 = match_op(op_5, z_val_2, z_val_3)
        z_val   = match_op(op_6, z_val_4, z_val_5)

        # create the final value tensors for the graph
        # if testing is enabled, the interim results will not be provided
        if testing:
            x = torch.tensor([[x_val], [-1.], [y_val], [-1],
                              [x_val], [-1.], [y_val], [-1],
                              [x_val], [-1.], [y_val], [-1],
                              [x_val], [-1.], [y_val], [-1],
                              [-1.], [-1], [-1.], [-1],
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
        # add input and output variables to the dataset
        data.x_val = x_val
        data.y_val = y_val
        data.z_val = z_val
        # add the expression as a string to the dataset
        data.expr_str = f"(({x_val.item()} {expr_dict[op_0]} {y_val.item()}) {expr_dict[op_4]} " \
                        f"({x_val.item()} {expr_dict[op_1]} {y_val.item()})) {expr_dict[op_6]} " \
                        f"(({x_val.item()} {expr_dict[op_2]} {y_val.item()}) {expr_dict[op_5]} " \
                        f"({x_val.item()} {expr_dict[op_3]} {y_val.item()}))"

        # add the expression without actual values as a string to the dataset
        data.expr_str_orig = f"((x {expr_dict[op_0]} y) {expr_dict[op_4]} " \
                             f"(x {expr_dict[op_1]} y)) {expr_dict[op_6]} " \
                             f"((x {expr_dict[op_2]} y) {expr_dict[op_5]} " \
                             f"(x {expr_dict[op_3]} y))"
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.num_classes = len(operation_dict)

        yield data


def match_op(op, x_val, y_val):
    """
    Helper function to match the chosen operation and calculate the corresponding next value
    """
    match op:
        case 0: return x_val + y_val
        case 1: return x_val - y_val
        case 2: return x_val * y_val
        case 3: return x_val & y_val
        case 4: return x_val | y_val
        case 5: return x_val ^ y_val
        case 6: return x_val >> y_val
        case 7: return x_val << y_val
        case _: return 0
