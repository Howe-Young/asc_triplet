from data_manager.datasets import DevSet
from data_manager.datasets_wrapper import BalanceBatchSampler
from data_manager.mean_variance import TaskbStandarizer
from data_manager.data_prepare import Dcase18TaskbData
from torchvision.transforms import Compose
from data_manager.transformer import ToTensor, Normalize
from torch.utils.data import DataLoader
import os
import networks
from losses import OnlineTripletLoss
import utils
from metrics import AverageNoneZeroTripletsMetric
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import fit


def hard_triplet_baseline_exp(device='3', lr=1e-3, n_epochs=300, n_classes=10, n_samples=12, margin=0.3, log_interval=50):
    """

    :param device:
    :param lr:
    :param n_epochs:
    :param n_classes:
    :param n_samples:
    :return:
    """

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
    loss_fn = OnlineTripletLoss(margin=margin, triplet_selector=utils.RandomNegativeTripletSelector(margin=margin))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5)

    fit(train_loader=train_batch_loader, val_loader=test_batch_loader, model=model, loss_fn=loss_fn,
        optimizer=optimizer, scheduler=scheduler, n_epochs=n_epochs, log_interval=log_interval,
        metrics=[AverageNoneZeroTripletsMetric()])

    train_embedding_tl, train_labels_tl = utils.extract_embeddings(train_batch_loader, model)
    utils.plot_embeddings(train_embedding_tl, train_labels_tl)
    test_embedding_tl, test_labels_tl = utils.extract_embeddings(test_batch_loader, model)
    utils.plot_embeddings(test_embedding_tl, test_labels_tl)

if __name__ == '__main__':

    kwargs = {
        'device': '2',
        'lr': 1e-3,
        'n_epochs': 50,
        'n_classes': 10,
        'n_samples': 12,
        'margin': 1.0,
        'log_interval': 80
    }

    hard_triplet_baseline_exp(**kwargs)
