"""
some loss class

"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.utilities import *


class OnlineTripletLoss(nn.Module):
    """
    online triplet loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and labels and return indices of triplets.
    """
    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
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


class HardTripletLoss(nn.Module):
    """
    Batch_Hard or Batch_All Triplet Loss
    """
    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        :param margin: margin for triplet loss
        :param hardest: if True, (Batch Hard) loss is considered only hardest triplets.
                        if False, (Batch All) loss is considerd all possible hard triplets.
        :param squared: if True, output is the pairwise squared euclidean distance matrix.
                        if False, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """

        :param embeddings: tensor of shape (batch_size, embed_dim)
        :param labels: tensor of shape (batch_size, )
        :return: triplet_loss and number of triplets
        """

        pairwise_dist = pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:

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
            num_hard_triplets = len(triplet_loss)
            triplet_loss = torch.mean(triplet_loss)

        else:
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
