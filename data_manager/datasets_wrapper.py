from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_manager.data_prepare import Dcase18TaskbData
from torch.utils.data.sampler import BatchSampler
from data_manager.datasets import DevSet
import numpy as np

"""
tripletDataset class: wrapper for Taskb_Development_set-like dataset, return random triplets(anchor, positive and negative)
"""

class TripletDevSet(Dataset):
    """
    triplets wrapper, return triplets
    """
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


class BalanceBatchSampler(BatchSampler):
    """
    batch sampler, randomly select n_classes, and n_samples each class
    """
    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels))
        self.labels_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.labels_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        self.dataset = dataset

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.labels_to_indices[class_][self.used_label_indices_count[class_] :
                                                              self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.labels_to_indices[class_]):
                    np.random.shuffle(self.labels_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size


class DatasetWrapper(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        data_, label_ = self.data[index], self.labels[index]
        if self.transform:
            data_, label_ = self.transform((data_, label_))
        return data_, label_

    def __len__(self):
        return len(self.labels)


# if __name__ == '__main__':
    # triplet_dataset = TripletDevSet(mode='train', device='b')
    # triplet_loader = DataLoader(dataset=triplet_dataset, batch_size=128, shuffle=True, num_workers=1)
    # for batch_id, (data, label) in enumerate(triplet_loader):
    #     print('batch id: ', batch_id)
    #     print('triplet data size: ', data[0].size(), data[1].size(), data[2].size())
    #     print('every batch first three labels: ', label[0][0].item(), label[1][0].item(), label[2][0].item())
    # dev_set = DevSet(mode='train', device='b')
    # batch_sampler = BalanceBatchSampler(dataset=dev_set, n_classes=10, n_samples=6)
    # loader = DataLoader(dataset=dev_set, batch_sampler=batch_sampler, num_workers=1)
    # for batch_id, data in enumerate(loader):
    #     print('batch id: ', batch_id)
    #     print('data: ', data[1])
