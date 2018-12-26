from data_manager.datasets import DevSet
from data_manager.datasets_wrapper import BalanceBatchSampler
from data_manager.mean_variance import TaskbStandarizer
from data_manager.data_prepare import Dcase18TaskbData
from torchvision.transforms import Compose
from data_manager.transformer import ToTensor, Normalize
from torch.utils.data import DataLoader
import os
from networks import vggish_bn
from losses import OnlineTripletLoss
from utils import RandomNegativeTripletSelector
from metrics import AverageNoneZeroTripletsMetric
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import fit


def hard_triplet_baseline_exp(device='3', lr=1e-3, n_epochs=300, n_classes=10, n_samples=12, margin=0.3):
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


    model = vggish_bn()
    model.cuda()
    loss_fn = OnlineTripletLoss(margin=margin, triplet_selector=RandomNegativeTripletSelector(margin=margin))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5)

    log_interval = 20
    fit(train_loader=train_batch_loader, val_loader=test_batch_loader, model=model, loss_fn=loss_fn,
        optimizer=optimizer, scheduler=scheduler, n_epochs=n_epochs, log_interval=log_interval,
        metrics=[AverageNoneZeroTripletsMetric()])

if __name__ == '__main__':

    kwargs = {
        'device': '2',
        'lr': 1e-3,
        'n_epochs': 300,
        'n_classes': 10,
        'n_samples': 12,
        'margin': 0.3
    }

    hard_triplet_baseline_exp(**kwargs)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle='False', num_workers=1)



    # for batch_id, (data, label) in enumerate(train_batch_loader):
    #     print("batch id: ", batch_id)
    #     print("data: ", data.shape)
    #     print('label', label)