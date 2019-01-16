"""
some loss class

"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.utilities import *


class RandomHardTripletLoss(nn.Module):
    """
    online triplet loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and labels and return indices of triplets.
    """
    def __init__(self, margin, triplet_selector):
        super(RandomHardTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, labels):
        triplets = self.triplet_selector.get_triplets(embeddings, labels)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class BatchHardTripletLoss(nn.Module):
    """
    Batch_Hard Triplet Loss
    """
    def __init__(self, margin=0.1, squared=False):
        """
        :param margin: margin for triplet loss
        :param squared: if True, output is the pairwise squared euclidean distance matrix.
                        if False, output is the pairwise euclidean distance matrix.
        """
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, embeddings, labels):
        """

        :param embeddings: tensor of shape (batch_size, embed_dim)
        :param labels: tensor of shape (batch_size, )
        :return: triplet_loss and number of triplets
        """

        pairwise_dist = pairwise_distance(embeddings, squared=self.squared)

        # get the hardest positive pairs (they should have biggest distance)
        # First, get a mask for every valid positive (they should have same label)
        mask_anchor_positive = get_anchor_positive_triplet_mask(labels).float()

        # put to zero any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        valid_positive_dist = pairwise_dist * mask_anchor_positive

        # shape (batch_size, 1)
        hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

        # for each anchor, get the hardest negative (they should have smallest distance)
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size, 1)
        hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        # count number of hard triplets (where triplet_loss > 0)
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)

        triplet_loss = torch.mean(triplet_loss)

        return triplet_loss, num_hard_triplets


class BatchAllTripletLoss(nn.Module):
    """
    batch all triplet loss
    """
    def __init__(self, margin=0.1, squared=False):
        """
        :param margin: margin for triplet loss
        :param squared: if True, output is the pairwise squared euclidean distance matrix.
                        if False, output is the pairwise euclidean distance matrix.
        """
        super(BatchAllTripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, embeddings, labels):
        """

        :param embeddings: tensor of shape (batch_size, embed_dim)
        :param labels: tensor of shape (batch_size, )
        :return: triplet_loss and number of triplets
        """

        pairwise_dist = pairwise_distance(embeddings, squared=self.squared)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = pairwise_dist.unsqueeze(dim=2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = pairwise_dist.unsqueeze(dim=1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size(batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, pos=j, neg=k
        # Uses broadcasting where the 1st argument has shape(batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # put to zero the invalid triplets
        mask = get_triplet_mask(labels).float()
        triplet_loss = triplet_loss * mask

        # remove negative losses (i.e. the easy triplets)
        triplet_loss = F.relu(triplet_loss)

        # count number of hard triplets (where triplet_loss > 0)
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)

        triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)
        return triplet_loss, num_hard_triplets

class BatchAllWithOutlierTripletLoss(nn.Module):
    def __init__(self, margin=1.0, squared=False, kernel_width=None):
        super(BatchAllWithOutlierTripletLoss, self).__init__()

        self.margin = margin
        self.squared = squared
        # beta == 1. / delta ** 2
        self.kernel_width = kernel_width

    def forward(self, embeddings, labels):
        """

        :param embeddings: tensor of shape (batch_size, embed_dim)
        :param labels: tensor of shape (batch_size, )
        :return: triplet_loss and number of triplets
        """

        pairwise_dist = pairwise_distance(embeddings, squared=self.squared)

        with torch.no_grad():
            if self.kernel_width is None:
                delta, _ = pairwise_dist.topk(k=8, dim=1, largest=False)
                gamma = 1 / 2 * (delta[:, -1] ** 2 + 1e-16)
                # gaussian kernel, pairwise similarities
                kernel_matrix = torch.exp(- torch.mul(pairwise_dist ** 2, gamma.view(-1, 1)))
            else:
                gamma = 1 / 2 * (self.kernel_width ** 2 + 1e-16)
                # gaussian kernel, pairwise similarities
                kernel_matrix = torch.exp(- torch.mul(pairwise_dist ** 2, gamma))

            anchor_negative_mask = get_anchor_negative_triplet_mask(labels=labels).float()
            # (batch_size,), sum over anchor-negative similarities
            sum_neg = torch.sum(kernel_matrix * anchor_negative_mask, dim=1)
            sum_all = torch.sum(kernel_matrix, dim=1) - kernel_matrix.diag()
            # (batch_size,), ratio between anchor-negative distances and all distances for each anchor
            outlier_prob = sum_neg / (sum_all + 1e-16)
            outlier_prob.clamp_(min=1e-16, max=1-1e-16)

            # count outlier num
            logging.info('outlier prob >0.6 num {:3.0f}'.format(torch.sum(torch.gt(outlier_prob, 0.6).float()).item()))

            # (batch_size, 1, 1), ready for broadcasting
            inlier_prob = (1 - outlier_prob).view(-1, 1, 1)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = pairwise_dist.unsqueeze(dim=2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = pairwise_dist.unsqueeze(dim=1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size(batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, pos=j, neg=k
        # Uses broadcasting where the 1st argument has shape(batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # put to zero the invalid triplets
        mask = get_triplet_mask(labels).float()
        triplet_loss = triplet_loss * mask

        # remove negative losses (i.e. the easy triplets)
        triplet_loss = F.relu(triplet_loss)

        # count number of hard triplets (where triplet_loss > 0)
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        assert num_hard_triplets > 1e-16

        triplet_loss = torch.sum(triplet_loss * inlier_prob) / (num_hard_triplets + 1e-16)
        return triplet_loss, num_hard_triplets


class LargeMarginLoss(nn.Module):
    """
    Better to use large batch size
    """
    def __init__(self, margin=1.0, squared=False, kernel_width=1.0):
        super(LargeMarginLoss, self).__init__()
        self.margin = margin
        self.squared = squared
        # beta == 1. / delta ** 2
        self.kernel_width = kernel_width

    def forward(self, embeddings, labels):
        pairwise_dist = pairwise_distance(embeddings, squared=self.squared)

        with torch.no_grad():
            gamma = 1 / 2 * (self.kernel_width ** 2 + 1e-16)

            # calc nearest positive probability
            gamma_pairwise_dist = torch.mul(pairwise_dist, - gamma)
            anchor_positive_mask = get_anchor_positive_triplet_mask(labels)  # ByteTensor
            gamma_pairwise_dist.masked_fill_(anchor_positive_mask ^ 1, float('-inf'))
            nearest_positive_prob = F.softmax(gamma_pairwise_dist, dim=1)

            # nearest negative probability
            gamma_pairwise_dist2 = torch.mul(pairwise_dist, - gamma)
            anchor_negative_mask = get_anchor_negative_triplet_mask(labels)
            gamma_pairwise_dist2.masked_fill_(anchor_negative_mask ^ 1, float('-inf'))
            nearest_negative_prob = F.softmax(gamma_pairwise_dist2, dim=1)

        # (batch_size, )
        ap = torch.sum(pairwise_dist * nearest_positive_prob, dim=1)
        # (batch_size, )
        an = torch.sum(pairwise_dist * nearest_negative_prob, dim=1)

        hinge_loss = F.relu(ap - an + self.margin)

        # count number of hard triplets (where triplet_loss > 0)
        hard_triplets = torch.gt(hinge_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        return torch.mean(hinge_loss), num_hard_triplets
