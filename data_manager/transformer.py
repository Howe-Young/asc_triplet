import torch
import numpy as np
from data_manager.data_prepare import Dcase18TaskbData


class ToTensor(object):
    def __call__(self, sample):
        data, label = torch.from_numpy(sample[0]), torch.from_numpy(np.array(sample[1]))
        data, label = data.type(torch.FloatTensor), label.type(torch.LongTensor)
        return data, label


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        data = sample[0]
        data = np.transpose(data, [0, 2, 1])
        data = (data - self.mean) / self.std
        # transpose back to (batch, frequency, time)
        data = np.transpose(data, [0, 2, 1])

        return data, sample[1]