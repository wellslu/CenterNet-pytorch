import mlconfig
import torch
from torch import nn

from .hourglass import *
from .resnet50 import resnet50, resnet50_Decoder, resnet50_Head
from .mobilenet import MobileNet, mobilenet_Decoder, mobilenet_Head

@mlconfig.register
class CenterNet_MobileNet(nn.Module):
    def __init__(self, num_classes = 20, pretrained = False):
        super(CenterNet_MobileNet, self).__init__()
        self.backbone = MobileNet(num_classes=num_classes)
        self.decoder = mobilenet_Decoder(1024)
        self.head = mobilenet_Head(num_classes=num_classes, channel=64)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))

@mlconfig.register
class CenterNet_Resnet50(nn.Module):
    def __init__(self, num_classes = 20, pretrained = False):
        super(CenterNet_Resnet50, self).__init__()
        # 512,512,3 -> 16,16,2048
        self.backbone = resnet50(pretrained = pretrained)
        # 16,16,2048 -> 128,128,64
        self.decoder = resnet50_Decoder(2048)
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        self.head = resnet50_Head(channel=64, num_classes=num_classes)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))


@mlconfig.register
class CenterNet(nn.Module):
    def __init__(self, heads={'hm': 80, 'wh': 2, 'reg':2}, pretrained=False, num_stacks=2, n=5, cnv_dim=256, channels=[256, 256, 384, 384, 384, 512], modules = [2, 2, 2, 2, 2, 4]):
        super(CenterNet, self).__init__()
        if pretrained:
            raise ValueError("HourglassNet has no pretrained model")

        self.nstack    = num_stacks
        self.heads     = heads

        curr_dim = channels[0]

        self.pre = nn.Sequential(
                    convolution(3, 128, 7, stride=2),
                    residual(128, 256, 3, stride=2)
                ) 
        
        self.kps  = nn.ModuleList([
            kp_module(
                n, channels, modules
            ) for _ in range(num_stacks)
        ])

        self.cnvs = nn.ModuleList([
            convolution(curr_dim, cnv_dim, 3) for _ in range(num_stacks)
        ])

        self.inters = nn.ModuleList([
            residual(curr_dim, curr_dim, 3) for _ in range(num_stacks - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, 1, bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(num_stacks - 1)
        ])
        
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, 1, bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(num_stacks - 1)
        ])

        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    nn.Sequential(
                        convolution(cnv_dim, curr_dim, 3, with_bn=False),
                        nn.Conv2d(curr_dim, heads[head], 1)
                    )  for _ in range(num_stacks)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    nn.Sequential(
                        convolution(cnv_dim, curr_dim, 3, with_bn=False),
                        nn.Conv2d(curr_dim, heads[head], 1)
                    )  for _ in range(num_stacks)
                ])
                self.__setattr__(head, module)


        self.relu = nn.ReLU(inplace=True)

    def freeze_backbone(self):
        freeze_list = [self.pre, self.kps]
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        freeze_list = [self.pre, self.kps]
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, image):
        # print('image shape', image.shape)
        inter = self.pre(image)
        outs  = []

        for ind in range(self.nstack):
            kp  = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp)

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

            out = {}
            for head in self.heads:
                out[head] = self.__getattr__(head)[ind](cnv)
            outs.append(out)
        return outs