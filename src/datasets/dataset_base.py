from torch.utils.data import Dataset
from pathlib import Path


class DatasetBase(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str = 'train',
        epoch_scaling=1,
        tiny_dataset: bool = False,
    ):
        if len(dataset_dir.split(',')) == 1:
            self.dataset_dir = Path(dataset_dir).expanduser()
        else:
            self.dataset_dir = [Path(d).expanduser() for d in dataset_dir.split(',')]
        self.split = split
        self.epoch_scaling = epoch_scaling
        self.tiny_dataset = tiny_dataset

    @property
    def real_length(self):
        raise NotImplementedError("Implement this in the child class")

    def __len__(self):
        if self.split == 'train':
            return self.real_length * self.epoch_scaling
        else:
            return self.real_length

    def __getitem__(self, index):
        index = index % self.real_length
        return self.getitem(index=index)

    def getitem(self, index):
        raise NotImplementedError("Implement this in the child class")
