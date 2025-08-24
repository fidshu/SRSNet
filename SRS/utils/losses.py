import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BCEDiceLoss']

import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return (0.5 * bce + dice)
        
class SoftIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-5

#        print(pred.size())
#        print(target.size())
        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss

class FusionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.soft = SoftIoULoss()
        self.bcedice = BCEDiceLoss()

    def forward(self, pred, target):
        return (self.soft(pred,target)+self.bcedice(pred,target))/2

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
class SoftIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-5

#        print(pred.size())
#        print(target.size())
        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss