from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split, DataLoader


class TextDataset(Dataset):
    def __init__(self, list_tensors, eos=50256):
        eos_tensor = torch.tensor([eos])
        self.sequences = list_tensors
        self.targets = [torch.cat((seq[1:], eos_tensor)) for seq in list_tensors]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def get_trainval(list_tensors, eos=50256, train_size=0.8):
    dataset = TextDataset(list_tensors, eos)

    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset
