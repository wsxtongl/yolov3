import os
import random
import xml.etree.cElementTree as et
from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from kmeans import *
import torch
path =r"D:\VOCdevkit\VOC2012"
image_path = r"D:\VOCdevkit\VOC2012\JPEGImages"
anno = "Annotations"

def Square_Generated (image,image_box): # 创建一个函数用来产生所需要的正方形图片转化
    w, h = image.size  # 得到图片的大小

    new_image = Image.new('RGB', size=(max(w, h), max(w, h)),color= 'black')  # 创建新的一个图片，大小取长款中最长的一边，color决定了图片中填充的颜色
    le = max(w, h)
    length = int(abs(w - h)/2)  # 一侧需要填充的长度
    if w < h:
        box = (length, 0)
        new_image.paste(image, box)       #产生新的图片
        image_box[:,0]+=length
        image_box[:,2]+=length
    else:
        box =(0, length)
        new_image.paste(image, box)
        image_box[:,1] += length
        image_box[:,3] += length
    new_image = new_image.resize((416,416))
    image_box = (image_box * 416 /le).astype(np.int)
    cx = image_box[:,0]+((image_box[:,2] - image_box[:,0])/2).astype(np.int)
    cy = image_box[:,1]+((image_box[:,3] - image_box[:,1])/2).astype(np.int)
    w = image_box[:,2] - image_box[:,0]
    h = image_box[:,3] - image_box[:,1]
    return new_image,cx,cy,w,h
def read_xml():
    class_num = 0
    class_map = {}
    class_name = set([])
    list_name = []
    res_str = ''
    files = os.listdir(os.path.join(path,anno))
    length_files = len(files)
    #random.shuffle(files)
    train_txt = open("./data/train.txt", 'w')
    test_txt = open("./data/test.txt", 'w')
    for i,xml_file in enumerate(files):
        try:

            xml_path = os.path.join(path,anno,xml_file)
            tree = et.parse(xml_path)
            root = tree.getroot()
            filename = root.find("filename").text   #找到图片名
            res_str += filename
            objects = root.findall("object")
            objects_box = []
            image_class = []
            for obj in objects:
                name_class = obj.find("name").text
                xmin = int(obj.find("bndbox").find("xmin").text)
                ymin = int(obj.find("bndbox").find("ymin").text)
                xmax = int(obj.find("bndbox").find("xmax").text)
                ymax = int(obj.find("bndbox").find("ymax").text)
                obox = [xmin,ymin,xmax,ymax]
                objects_box.append(obox)
                image_class.append(name_class)

                a_len = len(class_name)

                class_name.add(name_class)

                b_len = len(class_name)

                if b_len > a_len:
                    class_num += 1
                    class_map[name_class] = class_num-1
                    list_name.append(name_class)

                    print(list_name)
            image= Image.open(os.path.join(image_path,filename))
            img,cx,cy,w,h = Square_Generated(image,np.array(objects_box))

            listt = []
            for k in range(len(w)):
                list_data = cx[k],(cy[k]),w[k],h[k],class_map[image_class[k]]
                listt.append(list_data)
            encoding_str_list = [str(i) for k in listt for i in k]
            encoding_str = ','.join(encoding_str_list)

            if i < int(length_files*0.9):
                train_txt.write(filename+","+ encoding_str+"\n")
            else:
                test_txt.write(filename + "," + encoding_str + "\n")
            #img.save('./data/image/{0}'.format(filename))
        except Exception as e:
            print("e")
            continue
def make_kmeans():
    with open("./data/train.txt", 'r') as f:
        anchor_txt = open("anchor.txt", 'w')
        bo_list = []
        data_lines = f.readlines()
        for line in data_lines:
            line = line.strip('\n')
            line = line.split(',')
            _boxes = np.array([float(x) for x in line[1:]])
            # _boxes = np.array(list(map(float, strs[1:])))
            index_box = len(_boxes) // 5
            boxes = np.split(_boxes, index_box)
            for bo in boxes:
                w, h = bo[2], bo[3]
                bo_list.append([w, h])

        data_np = np.array(bo_list)
        out = kmeans(data_np, 9)
        area_data = out[:,0]*out[:,1]
        data = out[np.argsort(area_data)]
        data_list1 = [str(i) for k in data[:3] for i in k]
        data_list2 = [str(i) for k in data[3:6] for i in k]
        data_list3 = [str(i) for k in data[6:9] for i in k]
        list1_str = ','.join(data_list1)
        list2_str = ','.join(data_list2)
        list3_str = ','.join(data_list3)
        anchor_txt.write(list3_str+'\n'+list2_str+'\n'+list1_str)
        print(out)
        print("Accuracy: {:.2f}%".format(avg_iou(data_np, out) * 100))

if __name__ == '__main__':
    xm = read_xml()
    #data = make_kmeans()

