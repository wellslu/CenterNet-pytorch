import math
import mlconfig

import torch
import torch.nn as nn
import torch.nn.functional as F

@mlconfig.register
class RegL1(nn.Module):

    def __init__(self):
        super(RegL1, self).__init__()

    def forward(self, pred, target, mask):
        #--------------------------------#
        #   计算l1_loss
        #--------------------------------#
        pred = pred.permute(0,2,3,1)
        expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,2)

        loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss