import torch


def monotonic_constraint(individual):
    """
    Just to assert lambda_i+1 >= lambda_i
    """
    lambda_i = individual["lambda_i"]
    for i in range(1, len(lambda_i)):
        lambda_i[i] = max(lambda_i[i], lambda_i[i - 1])
    return individual


def check_valid_ids(tensor_list, vocab_size):
    """
    funtion to checkout if an id is "valid", it has an embedding
    """
    list_out = []
    for tensor in tensor_list:
        if any(t >= vocab_size for t in tensor):
            continue
        else:
            list_out.append(tensor)
    return list_out


def truncate_ids(tensor_list, vocab_size):
    """
    truncate the ids
    """
    list_out = []
    for tensor in tensor_list:
        truncated_tensor = [min(t, vocab_size - 1) for t in tensor]
        list_out.append(torch.tensor(truncated_tensor))
    return list_out


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]
