import math
import mlconfig

import torch
import torch.nn as nn
import torch.nn.functional as F

@mlconfig.register
class FocalLoss(nn.Module):

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, target):
        pred = pred.permute(0, 2, 3, 1)

        #-------------------------------------------------------------------------#
        #   找到每张图片的正样本和负样本
        #   一个真实框对应一个正样本
        #   除去正样本的特征点，其余为负样本
        #-------------------------------------------------------------------------#
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        #-------------------------------------------------------------------------#
        #   正样本特征点附近的负样本的权值更小一些
        #-------------------------------------------------------------------------#
        neg_weights = torch.pow(1 - target, 4)

        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        #-------------------------------------------------------------------------#
        #   计算focal loss。难分类样本权重大，易分类样本权重小。
        #-------------------------------------------------------------------------#
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        #-------------------------------------------------------------------------#
        #   进行损失的归一化
        #-------------------------------------------------------------------------#
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss