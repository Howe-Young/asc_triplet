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
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def batch_all_total_with_knn_exp(device='0', ckpt_prefix='Run01', lr=1e-3, embedding_epochs=10, classify_epochs=100,
                                 n_classes=10, n_samples=12, batch_size=128, margin=0.3, log_interval=50,
                                 log_level="INFO", k=3, squared=False, embed_dims=64, embed_net='vgg'):
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
    log_file = '{}/ckpt/batch_all_total_with_knn_exp/{}.log'.format(ROOT_DIR, ckpt_prefix)
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(filename=log_file, level=getattr(logging, log_level.upper(), None))
    logging.info(str(kwargs))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    # get the mean and std of dataset train/a
    standarizer = TaskbStandarizer(data_manager=Dcase18TaskbData())
    mu, sigma = standarizer.load_mu_sigma(mode='train', device='abc')

    # get the normalized train dataset
    train_dataset = DevSet(mode='train', device='abc', transform=Compose([
        Normalize(mean=mu, std=sigma),
        ToTensor()
    ]))
    test_dataset = {}
    for device in ['a', 'b', 'c']:
        test_dataset[device] = DevSet(mode='test', device=device, transform=Compose([
            Normalize(mean=mu, std=sigma),
            ToTensor()
        ]))

    train_batch_sampler = BalanceBatchSampler(dataset=train_dataset, n_classes=n_classes, n_samples=n_samples)
    train_batch_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=1)

    test_loader = {}
    for device in ['a', 'b', 'c']:
        test_loader[device] = DataLoader(dataset=test_dataset[device], batch_size=batch_size, shuffle=False, num_workers=1)
        # test_batch_sampler[device] = BalanceBatchSampler(dataset=test_dataset[device], n_classes=n_classes,
        #                                                  n_samples=n_samples)
        # test_batch_loader[device] = DataLoader(dataset=test_batch_sampler[device],
        #                                        batch_sampler=test_batch_sampler[device], num_workers=1)
    # test_batch_sampler = BalanceBatchSampler(dataset=test_dataset, n_classes=n_classes, n_samples=n_samples)
    # test_batch_loader = DataLoader(dataset=test_dataset, batch_sampler=test_batch_sampler, num_workers=1)

    if embed_net == 'vgg':
        model = networks.vggish_bn()
    elif embed_net == 'shallow':
        model = networks.embedding_net_shallow()
    else:
        print("{} network doesn't exist.".format(embed_net))
        return
    model = model.cuda()
    loss_fn = HardTripletLoss(margin=margin, hardest=False, squared=squared)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5)

    train_hist = History(name='train/a')
    val_hist = {}
    val_hist['a'], val_hist['b'], val_hist['c'] = History(name='test/a'), History(name='test/b'), History(name='test/c')

    ckpter = CheckPoint(model=model, optimizer=optimizer, path='{}/ckpt/batch_all_total_with_knn_exp'.format(ROOT_DIR),
                        prefix=ckpt_prefix, interval=1, save_num=1)

    for epoch in range(1, embedding_epochs + 1):
        scheduler.step()
        train_loss, metrics = train_epoch(train_loader=train_batch_loader, model=model, loss_fn=loss_fn,
                                          optimizer=optimizer, log_interval=log_interval,
                                          metrics=[AverageNoneZeroTripletsMetric()])
        train_logs = {}
        train_logs['loss'] = train_loss
        for metric in metrics:
            train_logs[metric.name()] = metric.value()
        train_hist.add(logs=train_logs, epoch=epoch)

        for device in ['a', 'b', 'c']:
            test_acc = kNN(model=model, train_loader=train_batch_loader, test_loader=test_loader[device], k=k)
            test_logs = {'acc': test_acc}
            val_hist[device].add(logs=test_logs, epoch=epoch)

        train_hist.clear()
        train_hist.plot()
        logging.info('Epoch{:04d}, {:15}, {}'.format(epoch, train_hist.name, str(train_hist.recent)))

        for device in ['a', 'b', 'c']:
            val_hist[device].plot()
            logging.info('Epoch{:04d}, {:15}, {}'.format(epoch, val_hist[device].name, str(val_hist[device].recent)))

        ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=val_hist['b'].recent)

    # train classifier using learned embeddings.
    classify_model = networks.classifier()
    classify_model = classify_model.cuda()
    classify_loss_fn = nn.CrossEntropyLoss()
    classify_optimizer = optim.Adam(classify_model.parameters(), lr=lr)
    classify_scheduler = lr_scheduler.StepLR(optimizer=classify_optimizer, step_size=30, gamma=0.5)
    classify_train_hist = History(name='classify_train/a')
    classify_val_hist = {}
    for device in ['a', 'b', 'c']:
        classify_val_hist[device] = History(name=('classify_val/' + str(device)))
    classify_ckpter = CheckPoint(model=classify_model, optimizer=classify_optimizer,
                                 path='{}/ckpt/batch_all_total_with_knn_exp'.format(ROOT_DIR),
                                 prefix=ckpt_prefix, interval=1, save_num=1)
    # reload best embedding model
    best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'), exp='batch_all_total_with_knn_exp').select_best(run=ckpt_prefix).selected_ckpt
    model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])

    train_embedding, train_labels = extract_embeddings(train_batch_loader, model, embed_dims)
    test_embedding, test_labels = {}, {}
    for device in ['a', 'b', 'c']:
        test_embedding[device], test_labels[device] = extract_embeddings(test_loader[device], model, embed_dims)

    classify_train_dataset = DatasetWrapper(data=train_embedding, labels=train_labels, transform=ToTensor())
    classify_test_dataset = {}
    for device in ['a', 'b', 'c']:
        classify_test_dataset[device] = DatasetWrapper(data=test_embedding[device], labels=test_labels[device],
                                                       transform=ToTensor())
    classify_train_loader = DataLoader(dataset=classify_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    classify_test_loader = {}
    for device in ['a', 'b', 'c']:
        classify_test_loader[device] = DataLoader(dataset=classify_test_dataset[device], batch_size=batch_size, shuffle=False,
                                                  num_workers=1)

    fit(train_loader=classify_train_loader, val_loader=classify_test_loader, model=classify_model,
        loss_fn=classify_loss_fn, optimizer=classify_optimizer, scheduler=classify_scheduler, n_epochs=classify_epochs,
        log_interval=log_interval, metrics=[AccumulatedAccuracyMetric()], train_hist=classify_train_hist,
        val_hist=classify_val_hist, ckpter=classify_ckpter, logging=logging)


if __name__ == '__main__':

    kwargs = {
        'ckpt_prefix': 'Run01',
        'device': '0',
        'lr': 1e-3,
        'embedding_epochs': 1,
        'classify_epochs': 5,
        'n_classes': 10,
        'n_samples': 12,
        'batch_size': 128,
        'margin': 1.0,
        'log_interval': 80,
        'k': 3,
        'squared': False,
        'embed_dims': 64,
        'embed_net': 'vgg'
    }

    batch_all_total_with_knn_exp(**kwargs)
