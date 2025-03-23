import torch


def load_data(data_path, tokenizer, max_length):
    with open(data_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    tokenized = tokenizer.encode(text_data)

    sequences = [tokenized[i : i + max_length] for i in range(0, len(tokenized), max_length)]

    tensor_data = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

    return tensor_data


def check_valid_ids(tensor_list, vocab_size):
    list_out = []
    for tensor in tensor_list:
        if any(t >= vocab_size for t in tensor):
            continue
        else:
            list_out.append(tensor)
    return list_out
