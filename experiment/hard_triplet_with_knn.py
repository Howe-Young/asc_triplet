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
from utils.utilities import *
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def hard_triplet_with_knn_exp(device='3', ckpt_prefix='Run01', lr=1e-3, embedding_epochs=10, n_epochs=100, n_classes=10, n_samples=12,
                              margin=0.3, log_interval=50, log_level="INFO", k=3):
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
    kwargs = locals()
    log_file = '{}/ckpt/hard_triplet_with_knn_exp/{}.log'.format(ROOT_DIR, ckpt_prefix)
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
        optimizer=optimizer, scheduler=scheduler, n_epochs=embedding_epochs, log_interval=log_interval,
        metrics=[AverageNoneZeroTripletsMetric()])

    # verification(model=model)
    train_embedding_tl, train_labels_tl = extract_embeddings(train_batch_loader, model, 64)
    # plot_embeddings(embeddings=train_embedding_tl, targets=train_labels_tl, title='train set')
    test_embedding_tl, test_labels_tl = extract_embeddings(test_batch_loader, model, 64)
    # plot_embeddings(embeddings=test_embedding_tl, targets=test_labels_tl, title='test set')

    distance_matrix = get_distance_matrix2(test_embedding_tl, train_embedding_tl)
    sorted_index = np.argsort(distance_matrix, axis=1)
    predict_labels = []
    for i in range(len(test_embedding_tl)):
        class_cnt = np.zeros([10])
        k_neighbor = train_labels_tl[sorted_index[i]]
        for j in range(k):
            # print(k_neighbor[j])
            class_cnt[int(k_neighbor[j])] += 1
        predict_labels.append(np.max(class_cnt))
    # print("test_labels:", test_labels_tl, 'len: ', len(test_labels_tl))
    # print("predict_labels: ", predict_labels)
    # print('len', len(predict_labels))
    predict_labels = np.array(predict_labels)
    # print(type(test_labels_tl))
    test_acc = (test_labels_tl == predict_labels).sum() / len(test_labels_tl)
    print("Test Accuracy: ", test_acc)





if __name__ == '__main__':

    kwargs = {
        'ckpt_prefix': 'Run01',
        'device': '0',
        'lr': 1e-3,
        'embedding_epochs': 10,
        'n_epochs': 50,
        'n_classes': 10,
        'n_samples': 12,
        'margin': 1.0,
        'log_interval': 80,
        'k': 5
    }

    hard_triplet_with_knn_exp(**kwargs)
