import json
import logging
import math
import os
import random
import xml.etree.cElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ANCHORS_GROUP_416 = {
    13: [[238.0, 133.0], [173.0, 248.0], [313.0, 276.0]],
    26: [[98.0, 76.0], [61.0, 132.0], [108.0, 184.0]],
    52: [[14.0, 20.0], [32.0, 36.0], [40.0, 72.0]]
}

save_class_map_path = r'D:\dataset\yolo\overfitting\class_map_v3.txt'
save_train_path = r'data/train1.txt'
val_path = r'data/image'


def validate(draw_nine_box):
    with open(save_class_map_path, 'r') as class_read:
        content_json = class_read.read()
    class_dic = json.loads(content_json)
    class_map_dic = {}
    for key in class_dic.keys():
        val = class_dic[key]
        class_map_dic[val] = key
    with open(save_train_path, 'r') as fi:
        content = fi.read()
    contents = content.split('\n')
    # random.shuffle(contents)
    for x, line in enumerate(contents):
        if x == 10:  # 校验数量
            break
        strs = line.split()
        _boxes = np.array(float(x) for x in strs[1:])
        _boxes = np.array(list(map(float, strs[1:])))
        if len(_boxes) == 0:
            continue
        boxes = np.split(_boxes, len(_boxes) // 5)
        _img_path = os.path.join(val_path, strs[0])
        _img_data = Image.open(_img_path)

        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900']
        for c_i, box in enumerate(boxes):
            draw = ImageDraw.Draw(_img_data)
            n_x = math.ceil(box[1] - (box[3] / 2))
            n_y = math.ceil(box[2] - (box[4] / 2))
            color_str = 'fuchsia' if c_i > 7 else c[c_i]
            draw.rectangle((n_x, n_y, n_x + box[3], n_y + box[4]), width=2, outline=color_str)
            font = ImageFont.truetype("consola.ttf", 20, encoding="unic")
            draw.text((n_x, n_y), str(class_map_dic[box[0]]), color_str, font)
            draw.ellipse((box[1], box[2], box[1] + 8, box[2] + 8), fill=color_str, outline=color_str)
            if not draw_nine_box:
                continue
            # verify anchors
            c_2 = ['fuchsia', 'blue', 'yellow']
            color_count = 0
            for anchors_key in ANCHORS_GROUP_416.keys():
                anchors_wh = ANCHORS_GROUP_416[anchors_key]
                for anchor in anchors_wh:
                    a_n_x = math.ceil(box[1] - anchor[0] / 2)
                    a_n_y = math.ceil(box[2] - anchor[1] / 2)
                    draw.rectangle((a_n_x, a_n_y, a_n_x + anchor[0], a_n_y + anchor[1]), width=2,
                                   outline=c_2[color_count])
                    font = ImageFont.truetype("consola.ttf", 10, encoding="unic")  # 设置字体
                    draw.text((a_n_x, a_n_y), str(anchors_key), c[color_count], font)
                color_count += 1
            _img_data.show()
            print(_img_path)
            print("*****" * 10)


if __name__ == '__main__':
    validate(True)
