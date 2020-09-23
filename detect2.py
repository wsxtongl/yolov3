import torch
from model import MainNet
from torchvision import transforms
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import os
import utils
import numpy as np
import time
import cv2
import cfg1
import torch
import cfg
device = "cuda" if torch.cuda.is_available() else "cpu"
conf = 0.7
cls_nms = 0.3
model_path = './model_param1/40.t'
anchor_group = {
    13:[[152.0, 231.0], [260.0, 143.0], [310.0, 277.0]],
    26:[[45.0, 97.0], [123.0, 97.0], [82.0, 166.0]],
    52:[[15.0, 18.0], [28.0, 43.0], [68.0, 50.0]]

}

class Detector():
    def __init__(self):
        self.net = MainNet(len(cfg.name))
        self.net.to(device)
        self.net.load_state_dict(torch.load(model_path))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
    def detect(self,image):
        w,h = image.size
        img = self.transform(image)
        img.unsqueeze_(0)
        img = img.to(device)
        self.net.eval()
        output_13, output_26, output_52 = self.net(img)
        idxs_13, vecs_13 = self.filter(output_13, conf)
        boxes_13 = self.parse(idxs_13, vecs_13, 32, anchor_group[13])

        idxs_26, vecs_26 = self.filter(output_26, conf)
        boxes_26 = self.parse(idxs_26, vecs_26, 16, anchor_group[26])

        idxs_52, vecs_52 = self.filter(output_52, conf)
        boxes_52 = self.parse(idxs_52, vecs_52, 8, anchor_group[52])
        if boxes_13.shape[0] == 0:
            boxes_13 = boxes_13.reshape(-1,7)
        if boxes_26.shape[0] == 0:
            boxes_26 = boxes_26.reshape(-1, 7)
        if boxes_52.shape[0] == 0:
            boxes_52 = boxes_52.reshape(-1, 7)
        return np.vstack([boxes_13, boxes_26, boxes_52])

    def filter(self,output, clsm):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        output = output.cpu().data
        torch.sigmoid_(output[..., 0])  # 置信度加sigmoid激活
        torch.sigmoid_(output[..., 1:3])  # 中心点加sigmoid激活

        mask =  output[..., 0] > clsm
        idxs = mask.nonzero()
        vecs = output[mask]
        return idxs, vecs

    def parse(self, idxs, vecs, t, anchors):
        if idxs.size(0) == 0:
            return torch.Tensor([])
        anchors = torch.Tensor(anchors)


        n = idxs[:, 0]  # 所属的图片
        a = idxs[:, 3]  # 建议框
        conf = vecs[:, 0]  # 置信度

        # （索引值+偏移量）*416/13
        cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # 原图的中心点x

        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        name = vecs[:, 5:]

        if name.shape[0] == 0:
            name = name.reshape(-1)
        else:
            name = torch.argmax(name, dim=1).float()

        np_boxes = torch.stack([n.float(),conf,x1, y1, x2, y2,name], dim=1).numpy()
        nms = utils.nms(np_boxes,cls_nms,False)

        return nms

def video_show():
    detector = Detector()
    cap = cv2.VideoCapture("1.jpg")

    while True:
        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img)
            width, high = im.size
            x_w = width / 416
            y_h = high / 416
            cropimg = im.resize((416, 416))

            boxes = detector.detect(cropimg)

            for box in boxes:  # 多个框，没循环一次框一个人脸
                x0 = float(box[1])
                x1 = int(box[2]*x_w)
                y1 = int(box[3]*y_h)
                x2 = int(box[4]*x_w)
                y2 = int(box[5]*y_h)
                n = int(box[6])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))
                frame = cv2.putText(frame, cfg.name[n-1]+' '+str("%.2f"%x0), (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            end_time = time.time()
            print("time", end_time - start_time)
            cv2.imshow('a', frame)
        cv2.waitKey(0)
def Test():
    test_path = r"./data/train.txt"
    img_path = r"./data/image"
    with open(test_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line.strip('\n')
            data = line.split(',')
            img = Image.open(os.path.join(img_path, data[0]))
            dector = Detector()
            boxes = dector.detect(img)

            for box in boxes:  # 多个框，没循环一次框一个人脸
                x0 = float(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                x2 = int(box[4])
                y2 = int(box[5])
                n = int(box[6])

                draw = ImageDraw.Draw(img)
                ttfront = ImageFont.truetype('simhei.ttf', 20)  # 字体大小
                draw.text((x1, y1 - 30), str(cfg.name[n]) + ' ' + str("%.2f" % x0), fill=(255, 0, 0), font=ttfront)
                draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
            plt.imshow(img)
            plt.pause(1)
if __name__ == '__main__':
    #show = video_show()
    test = Test()

