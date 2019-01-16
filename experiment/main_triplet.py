from data_manager.datasets import DevSet
from data_manager.datasets_wrapper import *
from data_manager.mean_variance import TaskbStandarizer
from data_manager.data_prepare import Dcase18TaskbData
from torchvision.transforms import Compose
from data_manager.transformer import *
from torch.utils.data import DataLoader
import os
import networks
from losses import *
from utils.selector import *
from metrics import *
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import *
import torch.nn as nn
from verification import *
import logging
from utils.history import *
from utils.checkpoint import *
from utils.utilities import *
from experiment.xgb import xgb_cls
import configparser
import losses
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed():
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)


def config_logging(config):
    log_file = '{}/experiment/ckpt/{}/{}.log'.format(ROOT_DIR,
                                          config['MAIN']['experiment'],
                                          config['MAIN']['ckpt_prefix'])
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    with open(log_file, 'w') as f:
        config.write(f)

    logging.basicConfig(filename=log_file, level=getattr(logging, config['MAIN']['log_level'].upper(), None))


def cross_entropy_pretrain(config, model, train_dataset, test_dataset):

    model.set_classify(True)

    if config['CE_PRETRAIN']['saved_ckpt'] == 'None':
        train_loader = DataLoader(dataset=train_dataset, batch_size=int(config['CE_PRETRAIN']['batch_size']),
                                  shuffle=True, num_workers=1)

        test_loader = DataLoader(dataset=test_dataset, batch_size=int(config['CE_PRETRAIN']['batch_size']),
                                 shuffle=False, num_workers=1)

        pt_loss_fn = nn.CrossEntropyLoss()
        pt_optimizer = optim.Adam(model.parameters(),
                                  lr=float(config['CE_PRETRAIN']['lr']),
                                  weight_decay=float(config['CE_PRETRAIN']['l2']))
        pt_scheduler = lr_scheduler.StepLR(optimizer=pt_optimizer, step_size=30, gamma=0.5)
        pt_train_hist = History(name='pretrain_train/a')
        pt_val_hist = History(name='pretrain_test/a')
        pt_ckpter = CheckPoint(model=model,
                               optimizer=pt_optimizer,
                               path='{}/experiment/ckpt/{}'.format(ROOT_DIR, config['MAIN']['experiment']),
                               prefix=(config['MAIN']['ckpt_prefix'] + 'pretrain'), interval=1, save_num=1)

        for epoch in range(1, int(config['CE_PRETRAIN']['epochs']) + 1):
            pt_scheduler.step()
            train_loss, metrics = train_epoch(train_loader=train_loader, model=model, loss_fn=pt_loss_fn,
                                              optimizer=pt_optimizer, log_interval=80,
                                              metrics=[AccumulatedAccuracyMetric()])
            train_logs = {'loss': train_loss}
            for metric in metrics:
                train_logs[metric.name()] = metric.value()
            pt_train_hist.add(logs=train_logs, epoch=epoch)

            test_loss, metrics = test_epoch(val_loader=test_loader, model=model, loss_fn=pt_loss_fn,
                                            metrics=[AccumulatedAccuracyMetric()])
            test_logs = {'loss': test_loss}
            for metric in metrics:
                test_logs[metric.name()] = metric.value()
            pt_val_hist.add(logs=test_logs, epoch=epoch)

            pt_train_hist.clear()
            pt_train_hist.plot()
            pt_val_hist.plot()
            logging.info('Epoch{:04d}, {:15}, {}'.format(epoch, pt_train_hist.name, str(pt_train_hist.recent)))
            logging.info('Epoch{:04d}, {:15}, {}'.format(epoch, pt_val_hist.name, str(pt_val_hist.recent)))
            pt_ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=pt_val_hist.recent)

        best_pt_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'experiment/ckpt'), exp=config['MAIN']['experiment']) \
            .select_best(run=(config['MAIN']['ckpt_prefix'] + 'pretrain')).selected_ckpt
        model.load_state_dict(torch.load(best_pt_model_filename)['model_state_dict'])
        model.set_classify(False)
    else:
        ckpt_path = os.path.join(ROOT_DIR, 'ckpt', config['MAIN']['experiment'], config['CE_PRETRAIN']['saved_ckpt'])
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        model.set_classify(False)

    return model


def train_triplet(config, model, train_balanced_loader,  train_loader, test_loader):

    if config['EMBEDDING']['saved_ckpt'] == 'None':

        loss_fn = getattr(losses, config['EMBEDDING']['triplet_loss'])(margin=float(config['EMBEDDING']['margin']),
                                                                    squared=config['EMBEDDING'].getboolean('squared'))

        optimizer = optim.Adam(model.parameters(),
                               lr=float(config['EMBEDDING']['lr']),
                               weight_decay=float(config['EMBEDDING']['l2']))
        scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5)

        train_hist = History(name='train/a')
        val_hist = History(name='test/a')
        ckpter = CheckPoint(model=model,
                            optimizer=optimizer,
                            path='{}/experiment/ckpt/{}'.format(ROOT_DIR, config['MAIN']['experiment']),
                            prefix=config['MAIN']['ckpt_prefix'], interval=1, save_num=1)

        for epoch in range(1, int(config['EMBEDDING']['epochs']) + 1):
            scheduler.step()
            train_loss, metrics = train_epoch(train_loader=train_balanced_loader, model=model, loss_fn=loss_fn,
                                              optimizer=optimizer, log_interval=80,
                                              metrics=[AverageNoneZeroTripletsMetric()])
            train_logs = dict()
            train_logs['loss'] = train_loss
            for metric in metrics:
                train_logs[metric.name()] = metric.value()
            train_hist.add(logs=train_logs, epoch=epoch)

            test_acc = kNN(model=model, train_loader=train_loader, test_loader=test_loader, k=int(config['KNN']['k']))
            test_logs = {'acc': test_acc}
            val_hist.add(logs=test_logs, epoch=epoch)

            train_hist.clear()
            train_hist.plot()
            val_hist.plot()
            logging.info('Epoch{:04d}, {:15}, {}'.format(epoch, train_hist.name, str(train_hist.recent)))
            logging.info('Epoch{:04d}, {:15}, {}'.format(epoch, val_hist.name, str(val_hist.recent)))
            ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=val_hist.recent)

        # reload best embedding model
        best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'experiment/ckpt'), exp=config['MAIN']['experiment']).\
            select_best(run=config['MAIN']['ckpt_prefix']).selected_ckpt
        model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
    else:
        ckpt_path = os.path.join(ROOT_DIR, 'experiment/ckpt', config['MAIN']['experiment'], config['EMBEDDING']['saved_ckpt'])
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

    return model


def run(config_file='config/large_margin.cfg'):

    set_seed()

    config = configparser.ConfigParser()
    config.read(os.path.join(ROOT_DIR, config_file))

    config_logging(config)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['MAIN']['device'])

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

    model = getattr(networks, config['MAIN']['net'])()
    model = model.cuda()

    if config['CE_PRETRAIN'].getboolean('enable'):
        model = cross_entropy_pretrain(config, model, train_dataset, test_dataset)

    train_batch_sampler = BalanceBatchSampler(dataset=train_dataset,
                                              n_classes=int(config['EMBEDDING']['n_classes']),
                                              n_samples=int(config['EMBEDDING']['n_samples']))
    train_balanced_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=1)
    train_loader = DataLoader(dataset=train_dataset, batch_size=int(config['EMBEDDING']['batch_size']),
                              shuffle=False, num_workers=1)

    test_loader = DataLoader(dataset=test_dataset, batch_size=int(config['EMBEDDING']['batch_size']),
                             shuffle=False, num_workers=1)
    model = train_triplet(config, model, train_balanced_loader, train_loader, test_loader)

    # train_embedding, train_labels = extract_embeddings(train_loader, model, embed_dims)
    # test_embedding, test_labels = extract_embeddings(test_loader, model, embed_dims)
    #
    # xgb_cls(train_data=train_embedding, train_label=train_labels, val_data=test_embedding, val_label=test_labels,
    #         exp_dir=os.path.dirname('{}/ckpt/{}/{}.log'.format(ROOT_DIR,
    #                                       config['MAIN']['experiment'],
    #                                       config['MAIN']['ckpt_prefix'])))


if __name__ == '__main__':

    run('config/outlier_triplet.cfg')
