import torch
import numpy as np
import os
import torch.nn.functional as F


# visualization module
import matplotlib.pyplot as plt
asc_classes = ['0', '1', '2', '3', '4', '5', '6', '7','8', '9', '10', '11', '12', '13', '14']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#00ff7f', '#9400d3', '#3b3b3b', '#0000ee', '#bcd2ee']

from sklearn.manifold import TSNE


def plot_embeddings(embeddings, targets, cls_num=10, xlim=None, ylim=None, title=None):
    plt.figure(figsize=(10, 10))
    # TODO init ?
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(embeddings)

    for i in range(cls_num):
        inds = np.where(targets == i)[0]

        plt.scatter(result[inds, 0], result[inds, 1], c=colors[i])

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    if cls_num == 10:
        plt.legend(asc_classes[:10])
    else:
        plt.legend(asc_classes)
    if title:
        plt.title(title)
    plt.show()


def extract_embeddings(dataloader, model, k_dims):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros([len(dataloader.dataset), k_dims])
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embeddings(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
        return embeddings, labels


def get_distance_matrix2(matrix1, matrix2):
    """
    give two embeddings matrix, computing distance between matrix1 and matrix2 every pairs.
    :param matrix1: shape of numpy (number of sample, embedding dims)
    :param matrix2: shape of numpy (number of sample, embedding dims)
    :return: distance matrix
    """
    distance_matrix = -2 * np.dot(matrix1, np.transpose(matrix2)) + (matrix1 ** 2).sum(axis=1).reshape(-1, 1) + \
                      (matrix2 ** 2).sum(axis=1).reshape(1, -1)
    return distance_matrix


def pairwise_distance(embeddings, squared=False):
    """
    Compute the 2D matrix of distance between all the embeddings.
    :param embeddings: tensor of shape (batch_size, embed_dim)
    :param squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
    :return: pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = dot_product.diag()
    distances = square_norm.unsqueeze(1) - 2 * dot_product + square_norm.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
    return distances


def get_triplet_mask(labels):
    """
    return a 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    :param labels: shape of tensor (batch_size, )
    :return: 3D mask
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # check if labels[i] == labels[j] and labels[j] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ 1)

    mask = distinct_indices * valid_labels # combine the two masks

    return mask


def get_anchor_positive_triplet_mask(labels):
    """
    Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    :param labels: tensor of shape (batch_size, )
    :return: tensor of shape (batch_size, batch_size)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # check that i and j are distinct
    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    # combine the two masks
    mask = indices_not_equal * labels_equal

    return mask


def get_anchor_negative_triplet_mask(labels):
    """
    return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    :param labels: tensor of shape (batch_size, )
    :return: tensor of shape (batch_size, batch_size)
    """

    # check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1

    return mask


class Reporter(object):
    def __init__(self, ckpt_root, exp, ckpt_file=None):
        self.ckpt_root = ckpt_root
        self.exp_path = os.path.join(self.ckpt_root, exp)
        self.run_list = os.listdir(self.exp_path)
        self.selected_ckpt = None
        self.selected_epoch = None
        self.selected_log = None
        self.selected_run = None

    def select_best(self, run=""):

        """
        set self.selected_run, self.selected_ckpt, self.selected_epoch
        :param run:
        :return:
        """

        matched = []
        for fname in self.run_list:
            if fname.startswith(run) and fname.endswith('tar'):
                matched.append(fname)

        acc = []
        import re
        for s in matched:
            acc_str = re.search('acc_(.*)\.tar', s).group(1)
            acc.append(float(acc_str))

        best_idx = np.argmax(acc)
        best_fname = matched[best_idx]

        self.selected_run = best_fname.split(',')[0]
        self.selected_epoch = int(re.search('Epoch_(.*),acc', best_fname).group(1))

        ckpt_file = os.path.join(self.exp_path, best_fname)

        self.selected_ckpt = ckpt_file

        return self
