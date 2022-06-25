import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, class_weights, reduction="none", ignore_index=-100):
        balanced_weights = nn.Parameter(torch.tensor(class_weights), requires_grad=False) if class_weights is not None else None
        self.fct = nn.CrossEntropyLoss(
            weight=balanced_weights,
            reduction=reduction,
            ignore_index=ignore_index,
        )

    def forward(self, x, target):
        return self.fct(x, target)


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, x, target):
        """
        x: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(x, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss
