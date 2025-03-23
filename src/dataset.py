from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    def __init__(self, list_tensors, eos=50256):
        eos_tensor = torch.tensor([eos])
        self.sequences = list_tensors
        self.targets = [torch.cat((seq[1:], eos_tensor)) for seq in list_tensors]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
