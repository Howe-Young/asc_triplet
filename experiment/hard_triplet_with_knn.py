from data_manager.datasets import DevSet
from data_manager.datasets_wrapper import *
from data_manager.mean_variance import TaskbStandarizer
from data_manager.data_prepare import Dcase18TaskbData
from torchvision.transforms import Compose
from data_manager.transformer import *
from torch.utils.data import DataLoader
import os
import networks
from losses import OnlineTripletLoss
from utils.selector import *
from metrics import *
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import fit
import torch.nn as nn
from verification import *
import logging
from utils.history import *
from utils.checkpoint import *
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def hard_triplet_baseline_exp(device='3', ckpt_prefix='Run01', lr=1e-3, n_epochs=300, n_classes=10, n_samples=12,
                              margin=0.3, log_interval=50, log_level="INFO"):
    """

    :param device:
    :param lr:
    :param n_epochs:
    :param n_classes:
    :param n_samples:
    :return:
    """
    kwargs = locals()
    log_file = '{}/ckpt/hard_triplet_baseline_exp/{}.log'.format(ROOT_DIR, ckpt_prefix)
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(filename=log_file, level=getattr(logging, log_level.upper(), None))
    logging.info(str(kwargs))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    # get the mean and std of dataset train/a
    standarizer = TaskbStandarizer(data_manager=Dcase18TaskbData())
    mu, sigma = standarizer.load_mu_sigma(mode='train', device='a')

    # get the normalized train dataset
    train_dataset = DevSet(mode='train', device='a', transform=Compose([
        Normalize(mean=mu, std=sigma),
        ToTensor()
    ]))
    test_dataset = DevSet(mode='test', device='a', transform=Compose([
        Normalize(mean=mu, std=sigma),
        ToTensor()
    ]))

    train_batch_sampler = BalanceBatchSampler(dataset=train_dataset, n_classes=n_classes, n_samples=n_samples)
    train_batch_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=1)

    test_batch_sampler = BalanceBatchSampler(dataset=test_dataset, n_classes=n_classes, n_samples=n_samples)
    test_batch_loader = DataLoader(dataset=test_dataset, batch_sampler=test_batch_sampler, num_workers=1)

    model = networks.embedding_net_shallow()
    model = model.cuda()
    loss_fn = OnlineTripletLoss(margin=margin, triplet_selector=RandomNegativeTripletSelector(margin=margin))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5)

    fit(train_loader=train_batch_loader, val_loader=test_batch_loader, model=model, loss_fn=loss_fn,
        optimizer=optimizer, scheduler=scheduler, n_epochs=n_epochs, log_interval=log_interval,
        metrics=[AverageNoneZeroTripletsMetric()])

    verification(model=model)
    train_embedding_tl, train_labels_tl = extract_embeddings(train_batch_loader, model, 64)
    # utils.plot_embeddings(embeddings=train_embedding_tl, targets=train_labels_tl, title='train set')
    test_embedding_tl, test_labels_tl = extract_embeddings(test_batch_loader, model, 64)
    # utils.plot_embeddings(embeddings=test_embedding_tl, targets=test_labels_tl, title='test set')

    model2 = networks.classifier()
    model2 = model2.cuda()
    loss_fn2 = nn.CrossEntropyLoss()
    optimizer2 = optim.Adam(model2.parameters(), lr=lr)
    scheduler2 = lr_scheduler.StepLR(optimizer=optimizer2, step_size=30, gamma=0.5)
    train_dataset2 = DatasetWrapper(data=train_embedding_tl, labels=train_labels_tl, transform=ToTensor())
    test_dataset2 = DatasetWrapper(data=test_embedding_tl, labels=test_labels_tl, transform=ToTensor())
    train_loader2 = DataLoader(dataset=train_dataset2, batch_size=128, shuffle=True, num_workers=1)
    test_loader2 = DataLoader(dataset=test_dataset2, batch_size=128, shuffle=False, num_workers=1)

    train_hist = History(name='train/a')
    val_hist = History(name='val/a')
    ckpter = CheckPoint(model=model, optimizer=optimizer, path='{}/ckpt/hard_triplet_baseline_exp'.format(ROOT_DIR),
                                  prefix=ckpt_prefix, interval=1, save_num=1)
    fit(train_loader=train_loader2, val_loader=test_loader2, model=model2, loss_fn=loss_fn2,
        optimizer=optimizer2, scheduler=scheduler2, n_epochs=n_epochs, log_interval=log_interval,
        metrics=[AccumulatedAccuracyMetric()], train_hist=train_hist, val_hist=val_hist, ckpter=ckpter, logging=logging)



if __name__ == '__main__':

    kwargs = {
        'device': '2',
        'lr': 1e-3,
        'n_epochs': 5,
        'n_classes': 10,
        'n_samples': 12,
        'margin': 1.0,
        'log_interval': 80
    }

    hard_triplet_baseline_exp(**kwargs)
