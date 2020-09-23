import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import math

img_path = r"data/image"
txt_path = r"data/train.txt"

img_lenth = 416
transform = transforms.Compose(
    [

        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))
    ]
)
# def one_hot(number,cls_num):
#     one_ho = np.zeros(cls_num)
#     one_ho[number] = 1
#     return one_ho
class Mydataset():
    def __init__(self):
        with open(txt_path,'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)
    def __getitem__(self, index):
        labels = {}

        lines = self.lines[index]
        data = lines.split(',')
        try:
            img = Image.open(os.path.join(img_path,data[0]))
            #img = Image.open(data[0])

            img = transform(img)
            boxes = np.array([float(k) for k in data[1:]])
            boxes = np.split(boxes,len(boxes)//5)
            anchor_dict = {}

            with open('anchor.txt','r') as f:
                anchor_list = []
                anchor_lines = f.readlines()
                for anchor in anchor_lines:
                    anchor = anchor.split(',')
                    anchor_np = np.array([float(k) for k in anchor])
                    anchor_list.append(anchor_np)
                anchor_dict[13] =[[anchor_list[0][0],anchor_list[0][1]],
                                  [anchor_list[0][2],anchor_list[0][3]],[anchor_list[0][4],anchor_list[0][5]]]
                anchor_dict[26] = [[anchor_list[1][0], anchor_list[1][1]],
                                   [anchor_list[1][2], anchor_list[1][3]], [anchor_list[1][4], anchor_list[1][5]]]
                anchor_dict[52] = [[anchor_list[2][0], anchor_list[2][1]],
                                   [anchor_list[2][2], anchor_list[2][3]], [anchor_list[2][4], anchor_list[2][5]]]
                anchor_area = {
                    13: [x * y for x, y in anchor_dict[13]],
                    26: [x * y for x, y in anchor_dict[26]],
                    52: [x * y for x, y in anchor_dict[52]],
                }
            for feature_size, anchors in anchor_dict.items():
                labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 6), dtype=np.float32)
                area_data = anchor_area[feature_size]
                for box in boxes:
                    cx,cy,w,h,clss = box

                    cx_offset, cx_index = math.modf(cx * feature_size / img_lenth)
                    cy_offset, cy_index = math.modf(cy * feature_size / img_lenth)
                    p_area = w * h

                    for i,row_anchor in enumerate(anchors):

                        w0,h0 = row_anchor[0],row_anchor[1]
                        inter = np.minimum(w, w0) * np.minimum(h, h0)  # 交集
                        conf = inter / (p_area + w0*h0 - inter)
                        iou = min(p_area, w0*h0) / max(p_area, w0*h0)
                        p_w, p_h = w / w0, h / h0
                        labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                            [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h),
                             int(clss)])

            return labels[13],labels[26], labels[52], img
        except Exception as e:
            print(e)
            
if __name__ == '__main__':
    dataset = Mydataset()
    print(dataset[0][0][0,0,0])