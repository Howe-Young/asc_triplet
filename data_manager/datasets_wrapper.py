from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_manager.data_prepare import Dcase18TaskbData
import numpy as np

"""
tripletDataset class: wrapper for Taskb_Development_set-like dataset, return random triplets(anchor, positive and negative)
"""

class TripletDevSet(Dataset):
    def __init__(self, mode='train', device='a', transform=None):
        self.mode = mode
        self.device = device
        self.transform = transform
        self.data_manager = Dcase18TaskbData()
        self.data, self.labels = self.data_manager.load_dev(mode=self.mode, devices=self.device)
        self.data = np.expand_dims(self.data, axis=1)

        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

        if mode == 'test':
            # generate fixed triplets for testing
            random_state = np.random.RandomState(29)
            triplets = [[i, random_state.choice(self.label_to_indices[i]),
                        random_state.choice(self.label_to_indices[
                            np.random.choice(list(self.labels_set - set([i])))
                                            ])]
                       for i in range(len(self.data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.mode == 'train':
            data1, label1 = self.data[index], self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            data2 = self.data[positive_index]
            data3 = self.data[negative_index]
            label2, label3 = self.labels[positive_index], self.labels[negative_index]
        else:
            data1 = self.data[self.test_triplets[index][0]]
            data2 = self.data[self.test_triplets[index][1]]
            data3 = self.data[self.test_triplets[index][2]]

        if self.transform is not None:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
            data3 = self.transform(data3)

        return (data1, data2, data3), (label1, label2, label3)

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    triplet_dataset = TripletDevSet(mode='train', device='b')
    triplet_loader = DataLoader(dataset=triplet_dataset, batch_size=128, shuffle=True, num_workers=1)
    for batch_id, (data, label) in enumerate(triplet_loader):
        print('batch id: ', batch_id)
        print('triplet data size: ', data[0].size(), data[1].size(), data[2].size())
        print('every batch first three labels: ', label[0][0].item(), label[1][0].item(), label[2][0].item())
