import math
import random
import cv2
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
# import imgaug.augmenters as iaa
# from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


# label form: dir/img x1,y1,x2,y2,class
class CenternetDataset(data.Dataset):
    def __init__(self, transform, annotation_lines, input_shape, num_classes, train=True):
        super(CenternetDataset, self).__init__()
        self.transform = transform
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)

        self.input_shape        = input_shape
        self.output_shape       = int(input_shape/4)
        self.num_classes        = num_classes
#         self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        line    = self.annotation_lines[index][:-1].split()
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
#         image   = np.array(Image.open(line[0]))
        image   = cv2.imread(line[0])
        image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ih, iw, _  = image.shape
        h, w    = self.input_shape, self.input_shape
        
        image = self.transform(image)
#         contrast_random = random.randrange(6, 15) / 10
#         color_random = random.randrange(6, 15) / 10
#         transform = transforms.Compose([
#                 transforms.ColorJitter(brightness=color_random),
#                 transforms.ColorJitter(contrast=contrast_random),
#             ])
#         image = transform(image)

        if len(box)>0:
            box[:, 0] = box[:, 0] / iw * w
            box[:, 2] = box[:, 2] / iw * w
            box[:, 1] = box[:, 1] / ih * h
            box[:, 3] = box[:, 3] / ih * h
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box


        batch_hm        = np.zeros((self.output_shape, self.output_shape, self.num_classes), dtype=np.float32)
        batch_wh        = np.zeros((self.output_shape, self.output_shape, 2), dtype=np.float32)
        batch_reg       = np.zeros((self.output_shape, self.output_shape, 2), dtype=np.float32)
        batch_reg_mask  = np.zeros((self.output_shape, self.output_shape), dtype=np.float32)
        
        if len(box) != 0:
            boxes = np.array(box[:, :4],dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / 4, 0, 127)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / 4, 0, 127)
            

        for i in range(len(box)):
            bbox    = boxes[i].copy()
            cls_id  = int(box[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                #-------------------------------------------------#
                #   计算真实框所属的特征点
                #-------------------------------------------------#
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                #----------------------------#
                #   绘制高斯热力图
                #----------------------------#
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                #---------------------------------------------------#
                #   计算宽高真实值
                #---------------------------------------------------#
                batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
                #---------------------------------------------------#
                #   计算中心偏移量
                #---------------------------------------------------#
                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                #---------------------------------------------------#
                #   将对应的mask设置为1
                #---------------------------------------------------#
                batch_reg_mask[ct_int[1], ct_int[0]] = 1

        return image, batch_hm, batch_wh, batch_reg, batch_reg_mask