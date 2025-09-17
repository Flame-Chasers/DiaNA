import torch
from torch.nn import functional as F


def compute_itc(sim_matrix, label_distribution):
    loss = -torch.sum(F.log_softmax(sim_matrix, dim=1) * label_distribution, dim=1).mean()
    return loss


def compute_sdm(sim_matrix, label_distribution, epsilon=1e-6):
    pred = F.softmax(sim_matrix, dim=1)
    loss = pred * (F.log_softmax(sim_matrix, dim=1) - torch.log(label_distribution + epsilon))
    loss = torch.mean(torch.sum(loss, dim=1))
    return loss
