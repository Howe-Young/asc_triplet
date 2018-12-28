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
from metrics import AverageNoneZeroTripletsMetric, AccumulatedAccuracyMetric
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import fit
import torch

def classification_baseline_exp(device='2', lr=1e-3, n_epochs=300, batch_size=128, log_interval=50):

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

    model = vggish_bn()
    model = model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5, last_epoch=-1)

    fit(train_loader=train_loader, val_loader=test_loader, model=model, loss_fn=loss_fn, optimizer=optimizer,
        scheduler=scheduler, n_epochs=n_epochs, log_interval=log_interval, metrics=[AccumulatedAccuracyMetric()])


if __name__ == '__main__':
    kwargs = {
        'device': 2,
        'lr': 1e-3,
        'n_epochs': 300,
        'batch_size': 128,
        'log_interval': 30
    }

    classification_baseline_exp(**kwargs)