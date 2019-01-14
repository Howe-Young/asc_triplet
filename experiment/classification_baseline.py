from data_manager.datasets import DevSet
from data_manager.mean_variance import TaskbStandarizer
from data_manager.data_prepare import Dcase18TaskbData
from torchvision.transforms import Compose
from data_manager.transformer import ToTensor, Normalize
from torch.utils.data import DataLoader
import os
from networks import vggish_bn
from metrics import AccumulatedAccuracyMetric
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import fit
import torch
from utils.history import *
from utils.checkpoint import *
import logging
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def classification_baseline_exp(device='2', ckpt_prefix='Run01', lr=1e-3, n_epochs=300, batch_size=128, log_interval=50,
                                classify=True, log_level='INFO'):

    kwargs = locals()
    log_file = '{}/ckpt/classification_exp/{}.log'.format(ROOT_DIR, ckpt_prefix)
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(filename=log_file, level=getattr(logging, log_level.upper(), None))
    logging.info(str(kwargs))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    standarizer = TaskbStandarizer(data_manager=Dcase18TaskbData())
    mu, sigma = standarizer.load_mu_sigma(mode='train', device='a')

    train_dataset = DevSet(mode='train', device='a', transform=Compose([
        Normalize(mean=mu, std=sigma),
        ToTensor()
    ]))
    test_dataset = DevSet(mode='test', device='a', transform=Compose([
        Normalize(mean=mu, std=sigma),
        ToTensor()
    ]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    model = vggish_bn(classify)
    model = model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5, last_epoch=-1)

    train_hist = History(name='train/a')
    val_hist = History(name='val/a')
    ckpter = CheckPoint(model=model, optimizer=optimizer, path='{}/ckpt/classification_exp'.format(ckpt_prefix),
                        prefix=ckpt_prefix, interval=1, save_num=1)


    fit(train_loader=train_loader, val_loader=test_loader, model=model, loss_fn=loss_fn, optimizer=optimizer,
        scheduler=scheduler, n_epochs=n_epochs, log_interval=log_interval, metrics=[AccumulatedAccuracyMetric()],
        train_hist=train_hist, val_hist=val_hist, ckpter=ckpter, logging=logging)


if __name__ == '__main__':
    kwargs = {
        'device': 1,
        'ckpt_prefix': 'Run01',
        'lr': 1e-3,
        'n_epochs': 300,
        'batch_size': 128,
        'log_interval': 30,
        'classify': True,
        'log_level': 'INFO'
    }

    classification_baseline_exp(**kwargs)