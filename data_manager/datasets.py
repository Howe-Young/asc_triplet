import numpy as np
from torch.utils.data import Dataset
from data_manager.data_prepare import Dcase18TaskbData

"""
implement datasets class
"""

class DevSet(Dataset):
    def __init__(self, mode='train', device='abc', transform=None):
        super(DevSet, self).__init__()
        self.data_manager = Dcase18TaskbData()
        self.data, self.label = self.data_manager.load_dev(mode=mode, devices=device)
        self.data = np.expand_dims(self.data, axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = (self.data[index], self.label[index])
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