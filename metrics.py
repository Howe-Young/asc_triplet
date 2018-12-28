import numpy as np


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1] # why outputs[0]? Because it's a tuple, outputs[0] is a ndarray
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNoneZeroTripletsMetric(Metric):
    """
    Counts average number of nonzero triplets found in minibatches
    """
    def __init__(self):
        self.values = []
        self.total_triplets = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        self.total_triplets.append(loss[2])
        return self.value(), self.total()

    def reset(self):
        self.values = []
        self.total_triplets = []

    def value(self):
        return np.mean(self.values)

    def total(self):
        return np.mean(self.total_triplets)

    def name(self):
        return 'Average nonzero triplets'