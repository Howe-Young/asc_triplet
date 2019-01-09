import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import os


class History(object):
    """
    class to save `loss` and  `accuracy`.
    """

    def __init__(self, name=None):
        self.name = name
        self.epoch = []
        self.acc = []
        self.loss = []
        self.axes = []
        self.nonzerostriplets = []
        self.eer = []
        self.recent = None

    def add(self, logs, epoch):
        self.recent = logs
        self.epoch.append(epoch)
        if 'loss' in logs.keys():
            self.loss.append(logs['loss'])
        if 'acc' in logs.keys():
            self.acc.append(logs['acc'])
        if 'nonzeros' in logs.keys():
            self.nonzerostriplets.append(logs['nonzeros'])
        if 'eer' in logs.keys():
            self.eer.append(logs['eer'])

    def set_axes(self, axes=None):
        if axes:
            self.axes = axes
        # new figure and axis
        else:
            self.axes = []
            plt.figure()
            num = int((len(self.acc) != 0) + (len(self.nonzerostriplets) != 0) + (len(self.eer) != 0) +
                      (len(self.loss) != 0))

            for i in range(num):
                self.axes.append(plt.subplot(num, 1, i + 1))

    def _get_tick(self):
        tick_max = np.max(self.epoch)
        ticks_int = np.arange(0, tick_max, np.ceil(tick_max / 5))
        if max(ticks_int) != tick_max:
            ticks_int = np.append(ticks_int, tick_max)
        return ticks_int

    def plot(self, axes=None, show=True):
        """
        plot loss and acc in subplots
        :param axes:
        :param show:
        :return:
        """
        self.set_axes(axes=axes)
        ticks = self._get_tick()
        cnt = 0
        if len(self.loss) != 0:
            self.axes[cnt].plot(self.epoch, self.loss)
            self.axes[cnt].legend([self.name + '/loss'])
            self.axes[cnt].set_xticks(ticks)
            self.axes[cnt].set_xticklabels([str(e) for e in ticks])
            cnt += 1
        if len(self.acc) != 0:
            self.axes[cnt].plot(self.epoch, self.acc)
            self.axes[cnt].legend([self.name + '/acc'])
            self.axes[cnt].set_xticks(ticks)
            self.axes[cnt].set_xticklabels([str(e) for e in ticks])
            cnt += 1
        if len(self.nonzerostriplets) != 0:
            self.axes[cnt].plot(self.epoch, self.nonzerostriplets)
            self.axes[cnt].legend([self.name + '/Non-zero triplets'])
            self.axes[cnt].set_xticks(ticks)
            self.axes[cnt].set_xticklabels([str(e) for e in ticks])
            cnt += 1
        if len(self.eer) != 0:
            self.axes[cnt].plot(self.epoch, self.eer)
            self.axes[cnt].legend([self.name + '/EER'])
            self.axes[cnt].set_xticks(ticks)
            self.axes[cnt].set_xticklabels([str(e) for e in ticks])

        plt.show() if show else None

    def clc_plot(self, axes=None, show=True):
        """
        clear output before plot, using in jupyter notebook to dynamically plot.
        :param axes:
        :param show:
        :return:
        """
        clear_output(wait=True)
        self.plot(axes=axes, show=show)

    def clear(self):
        clear_output(wait=True)

