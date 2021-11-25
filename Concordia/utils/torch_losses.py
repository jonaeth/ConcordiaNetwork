import torch
import torch.nn.functional as F


def kl_divergence(predictions, targets):
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    return kl_loss(F.log_softmax(predictions, dim=1), targets)

def cross_entropy(predictions, targets):
    return F.cross_entropy(predictions, targets)

