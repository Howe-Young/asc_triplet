import torch
import os
import numpy as np
np.random.seed(0)


class CheckPoint(object):
    """
    Bind model and optimizer
    """

    def __init__(self, model=None, optimizer=None, path=None, prefix="", interval=5, save_num=3, verbose=True):
        self.verbose = verbose
        self.model = model
        self.optimizer = optimizer
        self.path = path
        self.prefix = prefix
        self.interval = interval
        self.save_num = save_num
        self._monitored = []
        self._path_list = []
        self.histories = None
        self._bind_history_objs()

        if not os.path.exists(path):
            os.makedirs(path)

    def _get_save_path(self, epoch,  monitor, loss_acc):
        # e.g. ckpt/Run01,BaseConv,Epoch_0,acc_0.3323.tar
        full_path = os.path.join(self.path, '{},{},Epoch_{},{}_{:.6f}.tar'.format(self.prefix, type(self.model).__name__,
                                                                                  epoch, monitor, loss_acc[monitor]))
        return full_path

    def save(self, epoch, monitor, loss_acc, save_path=None):
        """
        save a checkpoint
        :param epoch:
        :param monitor:
        :param loss_acc:
        :param save_path:
        :return:
        """
        full_path = save_path if save_path else self._get_save_path(epoch, monitor, loss_acc)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'rng_state': torch.cuda.get_rng_state(),
            'histories': self.histories,
            monitor: loss_acc[monitor]
        }, full_path)
        print('[CheckPoint:]saved model to', full_path) if self.verbose else None

    def _delete_and_save(self, epoch, monitor, loss_acc, delete_idx):
        """
        delete old saved checkpoint and save new.
        :param epoch:
        :param monitor:
        :param loss_acc:
        :param delete_idx:
        :return:
        """
        old_path = self._path_list[delete_idx]
        if os.path.exists(old_path):
            os.remove(old_path)
            print('[CheckPoint:]delete file', old_path) if self.verbose else None
        else:
            print('[CheckPoint:]Fail to find and delete file', old_path) if self.verbose else None

        new_path = self._get_save_path(epoch, monitor, loss_acc)
        self._path_list[delete_idx] = new_path
        self.save(epoch, monitor, loss_acc, save_path=new_path)

    def _bind_history_objs(self):
        import gc
        from utils.history import History
        hist_list = []
        for obj in gc.get_objects():
            if isinstance(obj, History):
                hist_list.append(obj)

        self.histories = hist_list

    def bind_histories(self, hist_list=None):
        if hist_list:
            self.histories = hist_list
        else:
            self._bind_history_objs()

    def check_on(self, epoch, monitor, loss_acc):
        """
        save checkpoint if monitored value improve.
        :param epoch:
        :param monitor:
        :param loss_acc:
        :return:
        """
        if epoch % self.interval == 0:
            if len(self._monitored) < self.save_num:
                self._monitored.append(loss_acc[monitor])
                self._path_list.append(self._get_save_path(epoch, monitor, loss_acc))
                self.save(epoch, monitor, loss_acc)
            else:
                if monitor == 'loss':
                    if loss_acc['loss'] < max(self._monitored):
                        max_id = np.argmax(self._monitored)
                        self._monitored[int(max_id)] = loss_acc['loss']
                        self._delete_and_save(epoch, monitor, loss_acc, delete_idx=int(max_id))
                    else:
                        print('[CheckPoint:]loss not declined on {:.6f}'.format(min(self._monitored))) if self.verbose else None
                elif monitor == 'acc':
                    if loss_acc['acc'] > min(self._monitored):
                        min_id = np.argmin(self._monitored)
                        self._monitored[int(min_id)] = loss_acc['acc']
                        self._delete_and_save(epoch, monitor, loss_acc, delete_idx=int(min_id))
                        print("created")
                    else:
                        print('[CheckPoint:]acc not improved on {:.3f}'.format(max(self._monitored))) if self.verbose else None
                else:
                    print('[CheckPoint:]not supported CheckPoint monitor value') if self.verbose else None