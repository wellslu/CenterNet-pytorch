import mlconfig
from torch import nn

def nearby_int(x):
    return int(round(x))

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ConvBNReLU, self).__init__(*layers)

class DepthwiseConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride=1):
        layers = [
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(DepthwiseConv, self).__init__(*layers)

class MobileNet(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0, shallow=False):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(*self.make_layers(width_mult, shallow))
#         self.avg_pool = nn.AvgPool2d(7, stride=1)
#         self.classifier = nn.Linear(nearby_int(width_mult * 1024), num_classes)

    def forward(self, x):
        x = self.features(x)
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
        return x

    @staticmethod
    def make_layers(width_mult=1.0, shallow=False):
        settings = [
            (32, 2),
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
        ]
        if not shallow:
            settings += [(512, 1)] * 5
        settings += [
            (1024, 2),
            (1024, 1),
        ]

        layers = []
        in_channels = 3
        for i, (filters, stride) in enumerate(settings):
            out_channels = nearby_int(width_mult * filters)
            if i == 0:
                layers += [ConvBNReLU(in_channels, out_channels, stride=stride)]
            else:
                layers += [DepthwiseConv(in_channels, out_channels, stride=stride)]
            in_channels = out_channels
        return layers

class mobilenet_Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(mobilenet_Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False
        
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class mobilenet_Head(nn.Module):
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1):
        super(mobilenet_Head, self).__init__()
        # 热力图预测部分
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))

        # 中心点预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        return hm, wh, offset