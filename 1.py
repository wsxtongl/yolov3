from PIL import Image,ImageDraw
import os
import numpy as np
import math
import matplotlib.pyplot as plt

ANCHORS_GROUP_416 = {
    13: [[239.0,131.0],[173.0,245.0],[319.0,274.0]],
    26: [[98.0,74.0],[60.0,129.0],[107.0,180.0]],
    52: [[14.0,20.0],[33.0,34.0],[40.0,69.0]],

}
def a():
    test_path = r"./data/train1.txt"
    img_path = r"data\image"
    with open(test_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line.strip('\n')
            line = line.split(',')
            img = Image.open(os.path.join(img_path, line[0]))
            _boxes = np.array([float(x) for x in line[1:]])
            # _boxes = np.array(list(map(float, strs[1:])))
            index_box = len(_boxes) // 5
            boxes = np.split(_boxes, index_box)
            draw = ImageDraw.Draw(img)
            for box in boxes:
                for anchors_key in ANCHORS_GROUP_416.keys():
                    anchors_wh = ANCHORS_GROUP_416[anchors_key]
                    for anchor in anchors_wh:
                        a_n_x = math.ceil(box[1] - anchor[0] / 2)
                        a_n_y = math.ceil(box[2] - anchor[1] / 2)
                        draw.rectangle((a_n_x, a_n_y, a_n_x + anchor[0], a_n_y + anchor[1]), width=2,
                                       outline='red')


                        plt.imshow(img)
                        plt.pause(1)
if __name__ == '__main__':
    ab = a()

