import numpy as np
from torch.utils.data import Dataset
from data_manager.data_prepare import Dcase18TaskbData
from data_manager.dcase17_manager import Dcase17Data
from data_manager.dcase17_stdrizer import Dcase17Standarizer

"""
implement datasets class
"""


class DevSet(Dataset):
    """
    specify mode and device, return the after transform dataset.
    mode: train or test
    device: subset of abc(e.g. bc)
    transform: callable class
    """
    def __init__(self, mode='train', device='abc', transform=None):
        super(DevSet, self).__init__()
        self.data_manager = Dcase18TaskbData()
        self.data, self.labels = self.data_manager.load_dev(mode=mode, devices=device)
        self.data = np.expand_dims(self.data, axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = (self.data[index], self.labels[index])
        if self.transform:
            sample = self.transform(sample)
        return sample


class d17DevSet(Dataset):
    def __init__(self, mode='train', fold_idx=1, transform=None):
        super(d17DevSet, self).__init__()
        self.standarizer = Dcase17Standarizer(data_manager=Dcase17Data())
        self.data, self.labels = self.standarizer.load_dev_standrized(fold_idx=fold_idx, mode=mode)
        self.data = np.expand_dims(self.data, axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = (self.data[index], self.labels[index])
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_set = DevSet(mode='train', device='a')
    data_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True, num_workers=1)
    for batch_id, (data, label) in enumerate(data_loader):
        print('batch id: ', batch_id)
        print('data: ', data)
        print('label: ', label)