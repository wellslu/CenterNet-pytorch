from src.utils import decode_bbox
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
import argparse
import time
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img', type=str, default='./img.jpg')
    parser.add_argument('-p', '--predict', type=str, default='./img_predict.jpg')
    return parser.parse_args()


def main():
    args = parse_args()
    model = torch.load('./model/best_loss.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512,512)),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
    img = cv2.imread(args.img)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start_predict = time.time()
    img1 = transform(img1)
    img1 = img1.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img1.unsqueeze(0))
    hm = outputs[0].to('cpu')
    wh = outputs[1].to('cpu')
    reg = outputs[2].to('cpu')
    outputs = decode_bbox(hm, wh, reg, 0.5, True)
    boxes = []
    scores = []
    cls = []
    if outputs[0] != []:
        for output in outputs[0]:
            boxes.append(output[:4].numpy())
            scores.append(output[4])
            cls.append(output[5])
    boxes, scores, cls = np.array(boxes), np.array(scores), np.array(cls)
    end_predict = time.time()
    if len(boxes) > 0:
        boxes[:, 0] = boxes[:, 0] * img.shape[1]
        boxes[:, 1] = boxes[:, 1] * img.shape[0]
        boxes[:, 2] = boxes[:, 2] * img.shape[1]
        boxes[:, 3] = boxes[:, 3] * img.shape[0]
        for box, score, cl in zip(boxes, scores, cls):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            score = round(score.item(), 2)
            cl = cl.item()
            cv2.putText(img, f'class : {cl}, point : {score}', (xmin, ymin), cv2.FONT_ITALIC, 3, (0, 255, 0), 2)
        cv2.imwrite(args.predict, img)
    else:
        cv2.imwrite(args.predict, img)
    print('time: ', round(end_predict-start_predict, 3))
if __name__ == '__main__':
    main()