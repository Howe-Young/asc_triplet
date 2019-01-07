import torch
import numpy as np


# visualization module
import matplotlib.pyplot as plt
asc_classes = ['0', '1', '2', '3', '4', '5', '6', '7','8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

from sklearn.manifold import TSNE


def plot_embeddings(embeddings, targets, xlim=None, ylim=None, title=None):
    plt.figure(figsize=(10, 10))
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(embeddings)

    for i in range(10):
        inds = np.where(targets == i)[0]

        plt.scatter(result[inds, 0], result[inds, 1], c=colors[i])

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
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
    :param matrix1: shape(number of sample, embedding dims)
    :param matrix2: shape(number of sample, embedding dims)
    :return: distance matrix
    """
    distance_matrix = -2 * np.dot(matrix1, np.transpose(matrix2)) + (matrix1 ** 2).sum(axis=1).reshape(-1, 1) + \
                      (matrix2 ** 2).sum(axis=1).reshape(1, -1)
    return distance_matrix
