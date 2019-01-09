from data_manager.datasets_wrapper import *
from data_manager.transformer import *
from data_manager.mean_variance import *
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import numpy as np
import networks
import matplotlib.pyplot as plt

def get_distance_matrix(vector):
    distance_matrix = -2.0 * np.dot(vector, np.transpose(vector)) + (vector ** 2).sum(axis=1).reshape(1, -1) + \
                      (vector ** 2).sum(axis=1).reshape(-1, 1)
    return distance_matrix


def verification(model, mode='test', device='a'):
    """
    get the equal error rate through verification set.
    :param model:
    :param mode:
    :param device:
    :return:
    """
    standarizer = TaskbStandarizer(data_manager=Dcase18TaskbData())
    mu, sigma = standarizer.load_mu_sigma(mode='train', device='a')
    test_dataset = DevSet(mode=mode, device=device, transform=Compose([
        Normalize(mean=mu, std=sigma),
        ToTensor()
    ]))

    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=1)
    total_embeddings = np.zeros([len(test_loader.dataset), 64])
    labels = np.zeros([len(test_loader.dataset)])
    k = 0
    with torch.no_grad():
        model.eval()
        for batch_id, (data, target) in enumerate(test_loader):
            data = data.cuda()
            total_embeddings[k:k+len(data)] = model.get_embeddings(data).data.cpu().numpy()
            labels[k:k+len(data)] = target.numpy()
            k += len(data)

    distance_matrix = get_distance_matrix(total_embeddings)
    total_distance = []
    total_target = []
    labels_set = set(test_dataset.labels)
    label_to_indices = {label: np.where(test_dataset.labels == label)[0] for label in labels_set}
    length = len(test_dataset.labels)
    vis = np.zeros([length, length])
    for i in range(length):
        vis[i][i] = 1

    for label in label_to_indices.keys():
        for sample1 in label_to_indices[label]:
            for sample2 in label_to_indices[label]:
                if vis[sample1][sample2] == 0:
                    total_distance.append(distance_matrix[sample1][sample2])
                    total_target.append(1)
                    vis[sample1][sample2] = 1
                    vis[sample2][sample1] = 1
            left_set = labels_set - set([label])
            for left_label in left_set:
                for sample2 in label_to_indices[left_label]:
                    if vis[sample1][sample2] == 0:
                        total_distance.append(distance_matrix[sample1][sample2])
                        total_target.append(0)
                        vis[sample1][sample2] = 1
                        vis[sample2][sample1] = 1

    eer = EER(total_distance, total_target, True)
    print('ERR is: ', eer)



def EER(distance, target, show_fig=False):
    """
    give distance list and true label(non-target or target), compute equal error rate.
    :param distance:
    :param target:
    :param show_fig:
    :return:
    """
    index = sorted(range(len(distance)), key=lambda k: distance[k])
    distance.sort()
    target = np.array(target)
    target = target[index]
    eps = 1e-6
    up_sum = np.ones([len(distance)])
    up_sum = np.cumsum(up_sum == 1)
    down_sum = np.ones([len(distance)]) * len(distance)
    down_sum = down_sum - up_sum
    down_sum[-1] += eps
    FA = np.cumsum(target == 0)
    target_reverse = target[::-1]
    FR = np.cumsum(target_reverse == 1)
    FR = FR[::-1]
    FAR = FA / up_sum
    FRR = FR / down_sum
    x = np.arange(0, len(FAR), 1)
    plt.figure()
    plt.plot(x, FAR)
    plt.title('FAR')
    plt.figure()
    plt.plot(x, FRR)
    plt.title("FRR")

    if show_fig:
        plt.figure()
        plt.plot(FRR, FAR)
        plt.show()
    diff = np.abs(FAR - FRR)
    EER_index = np.where(diff == np.min(diff))
    print('FAR: ', FAR[EER_index])
    print('FRR: ', FRR[EER_index])
    return FAR[EER_index]


from utils.utilities import *


def kNN(model, train_loader, test_loader, k=3):

    train_embedding, train_labels = extract_embeddings(train_loader, model, 128)
    test_embedding, test_labels = extract_embeddings(test_loader, model, 128)

    distance_matrix = get_distance_matrix2(test_embedding, train_embedding)
    sorted_index = np.argsort(distance_matrix, axis=1)
    predict_labels = []
    for i in range(len(test_embedding)):
        class_cnt = np.zeros([10])
        k_neighbor = train_labels[sorted_index[i]]
        for j in range(k):
            class_cnt[int(k_neighbor[j])] += 1
        predict_labels.append(np.argmax(class_cnt))
    predict_labels = np.array(predict_labels)
    # test_acc = (test_labels == predict_labels).sum() / len(test_labels)
    test_acc = np.mean(test_labels == predict_labels)
    return test_acc


if __name__ == '__main__':
    model = networks.embedding_net_shallow()
    model = model.cuda()
    verification(model=model)