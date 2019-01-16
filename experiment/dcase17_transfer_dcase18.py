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
from data_manager.dcase17_stdrizer import Dcase17Standarizer
from data_manager.dcase17_manager import Dcase17Data
from data_manager.datasets import *
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def triplet_loss_with_knn_exp(device='3', ckpt_prefix='Run01', lr=1e-3, dcase17_epochs=10, dcase18_epochs=100,
                           n_classes=10, n_samples=12, margin=0.3, log_interval=50, log_level="INFO", k=3,
                           squared=False, embed_dims=64, embed_net='vgg', batch_size=128, select_method='batch_all'):
    """
    knn as classifier.
    :param device:
    :param lr:
    :param n_epochs:
    :param n_classes:
    :param n_samples:
    :param k: kNN parameter
    :return:
    """
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    kwargs = locals()
    log_file = '{}/ckpt/transfer_{}_with_knn_exp/{}.log'.format(ROOT_DIR, select_method, ckpt_prefix)
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(filename=log_file, level=getattr(logging, log_level.upper(), None))
    logging.info(str(kwargs))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    # create dcase17 train and test dataset
    d17_train_dataset = d17DevSet(mode='train', fold_idx=1, transform=ToTensor())
    d17_test_dataset = d17DevSet(mode='test', fold_idx=1, transform=ToTensor())

    # build dcase17 train batch loader(for triplet loss training, drop last) and train loader(for kNN verification)
    d17_train_batch_sampler = BalanceBatchSampler(dataset=d17_train_dataset, n_classes=n_classes, n_samples=n_samples)
    d17_train_batch_loader = DataLoader(dataset=d17_train_dataset, batch_sampler=d17_train_batch_sampler, num_workers=1)
    d17_train_loader = DataLoader(dataset=d17_train_dataset, batch_size=batch_size, num_workers=1)

    # build dcase17 test loader
    d17_test_loader = DataLoader(dataset=d17_test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # get dcase18 train/A mean and variance
    d18_standarizer = TaskbStandarizer(data_manager=Dcase18TaskbData())
    mu, sigma = d18_standarizer.load_mu_sigma(mode='train', device='a')

    # get the normalized train dataset of dcase18
    d18_train_dataset = DevSet(mode='train', device='a', transform=Compose([
        Normalize(mean=mu, std=sigma),
        ToTensor()
    ]))

    # get the normalized test dataset of dcase18
    d18_test_dataset = DevSet(mode='test', device='a', transform=Compose([
        Normalize(mean=mu, std=sigma),
        ToTensor()
    ]))

    # build dcase18 train loader and test loader
    d18_train_loader = DataLoader(dataset=d18_train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    d18_test_loader = DataLoader(dataset=d18_test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # determine the network architecture
    if embed_net == 'vgg':
        model = networks.vggish_bn(classify=False)
    elif embed_net == 'shallow':
        model = networks.embedding_net_shallow()
    else:
        print("{} doesn't exist!".format(embed_net))
        return

    model = model.cuda()

    # the select method of triplets
    if select_method == 'batch_all':
        loss_fn = BatchAllTripletLoss(margin=margin, squared=squared)
    elif select_method == 'batch_hard':
        loss_fn = BatchHardTripletLoss(margin=margin, squared=squared)
    elif select_method == 'random_hard':
        loss_fn = RandomHardTripletLoss(margin=margin, triplet_selector=RandomNegativeTripletSelector(margin=margin))
    else:
        print("{} is not defined!".format(select_method))
        return

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5)

    # build history class record training process
    d17_train_hist = History(name='Dcase17_train/a')
    d17_val_hist = History(name='Dcase17_test/a')

    # build checkpoint
    d17_ckpter = CheckPoint(model=model, optimizer=optimizer, path='{}/ckpt/transfer_{}_with_knn_exp'.format(ROOT_DIR, select_method),
                        prefix=(ckpt_prefix + 'Dcase17'), interval=1, save_num=1)

    d18_train_hist = History(name='Dcase18_train/a')
    d18_val_hist = History(name='Dcase18_test/a')
    d18_ckpter = CheckPoint(model=model, optimizer=optimizer, path='{}/ckpt/transfer_{}_with_knn_exp'.format(ROOT_DIR, select_method),
                        prefix=(ckpt_prefix + 'Dcase18'), interval=1, save_num=1)

    # training process
    for epoch in range(1, dcase17_epochs + 1):
        scheduler.step()
        train_loss, metrics = train_epoch(train_loader=d17_train_batch_loader, model=model, loss_fn=loss_fn,
                                          optimizer=optimizer, log_interval=log_interval,
                                          metrics=[AverageNoneZeroTripletsMetric()])
        train_logs = dict()
        train_logs['loss'] = train_loss
        for metric in metrics:
            train_logs[metric.name()] = metric.value()
        d17_train_hist.add(logs=train_logs, epoch=epoch)

        d17_test_acc = kNN(model=model, train_loader=d17_train_loader, test_loader=d17_test_loader, k=k, cls_num=15)
        d17_test_logs = {'acc': d17_test_acc}
        d17_val_hist.add(logs=d17_test_logs, epoch=epoch)

        d18_test_acc = kNN(model=model, train_loader=d18_train_loader, test_loader=d18_test_loader, k=k, cls_num=10)
        d18_test_logs = {'acc': d18_test_acc}
        d18_val_hist.add(logs=d18_test_logs, epoch=epoch)

        d17_train_hist.clear()
        d17_train_hist.plot()
        d17_val_hist.plot()
        d18_val_hist.plot()
        logging.info('Epoch{:04d}, {:15}, {}'.format(epoch, d17_train_hist.name, str(d17_train_hist.recent)))
        logging.info('Epoch{:04d}, {:15}, {}'.format(epoch, d17_val_hist.name, str(d17_val_hist.recent)))
        logging.info('Epoch{:04d}, {:15}, {}'.format(epoch, d18_val_hist.name, str(d18_val_hist.recent)))

        # check and save best model
        d17_ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=d17_val_hist.recent)
        d18_ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=d18_val_hist.recent)

    # reload best embedding model
    best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'), exp='transfer_{}_with_knn_exp'.\
                                   format(select_method)).select_best(run=(ckpt_prefix+'Dcase17')).selected_ckpt
    model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])

    d17_train_embeddings, d17_train_labels = extract_embeddings(dataloader=d17_train_loader, model=model, k_dims=embed_dims)
    d17_test_embeddings, d17_test_labels = extract_embeddings(dataloader=d17_test_loader, model=model, k_dims=embed_dims)

    d18_train_embeddings, d18_train_labels = extract_embeddings(dataloader=d18_train_loader, model=model, k_dims=embed_dims)
    d18_test_embeddings, d18_test_labels = extract_embeddings(dataloader=d18_test_loader, model=model, k_dims=embed_dims)

    plot_embeddings(d17_train_embeddings, d17_train_labels, cls_num=15, title='train data embedding visualization')
    plot_embeddings(d17_test_embeddings, d17_test_labels, cls_num=15, title='test data embedding visualization')

    plot_embeddings(d18_train_embeddings, d18_train_labels, cls_num=10, title='train data embedding visualization')
    plot_embeddings(d18_test_embeddings, d18_test_labels, cls_num=10, title='test data embedding visualization')

# train_embedding, train_labels = extract_embeddings(train_batch_loader, model, embed_dims)
    # test_embedding, test_labels = extract_embeddings(test_loader, model, embed_dims)
    #
    # xgb_cls(train_data=train_embedding, train_label=train_labels, val_data=test_embedding, val_label=test_labels,
    #         exp_dir=os.path.dirname(log_file))

    # TODO plot all curve
    # if using_pretrain:
    #     pt_train_hist.plot()
    #     pt_val_hist.plot()


if __name__ == '__main__':

    kwargs = {
        'ckpt_prefix': 'Run01',
        'device': '0',
        'lr': 1e-3,
        'dcase17_epochs': 3,
        'dcase18_epochs': 3,
        'n_classes': 10,
        'n_samples': 12,
        'margin': 1.0,
        'log_interval': 80,
        'k': 3,
        'squared': False,
        'embed_dims': 64,
        'embed_net': 'vgg',
        'batch_size': 128,
        'select_method': 'batch_all'
    }

    triplet_loss_with_knn_exp(**kwargs)
