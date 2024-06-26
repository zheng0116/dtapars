import argparse
import xml.etree.ElementTree as ET
import os
from os import listdir,getcwd
from os.path import join
import random
from shutil import copyfile

classes = ['persion']
TRAIN_RATIO =90
parser = argparse.ArgumentParser(description="Convert xml to txt for yolo traing")
parser.add_argument("image_path",dest="file_path",default="",type=str)
parser.add_argument("label_path",dest="label_path",default="",type=str)

def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for file in dir_list:
        abspath = os.path.join(os.path.abspath(path),file)
        if os.path.isfile(abspath):
            if file.startswith("._"):
                os.remove(abspath)

'''
convert xml to txt(yolo):
tw:image width reciprocal th:image height reciprocal
cx: (xmin+ymin) /2、cy: (xman+ymax) /2
bw: (ymin-xmin)、bh: (ymax-xmax)
x: cx*tw、y: cy*th、w: bw*tw、h: bh*th
'''
def convert(size,box):
    tw = 1. / size[0]
    th = 1. / size[1]
    cx = (box[0] + box[1]) /2.0
    cy = (box[2] + box[3]) /2.0
    bw = box[1] - box[0]
    bh = box[3] - box[2]
    x = cx*tw
    y = cy*th
    w = bw*tw
    h = bh*th
    return (x,y,w,h)

def convert_label(image_id,out_file,label_file):
    labelfile = open(f"{label_file}/{image_id}.xml")
    output = open(f"{out_file}/{image_id}.txt","w")
    tree = ET.parse(labelfile)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        class_id = classes.index(cls)
        xmlbox = obj.find("binbox")
        b = (float(xmlbox.find("xmin").text),
             float(xmlbox.find("xmax").text),
             float(xmlbox.find("ymin").text),
             float(xmlbox.find("ymax").text)
        )
        size = convert((w,h),b)
        output.write(str(class_id)+" "+" ".join([str(a) for a in size ]))
    labelfile.close()
    output.close()
wd = os.getcwd()
wd = os.getcwd()
data_base_dir = os.path.join(wd, "C:/Users/zheng/Desktop/yolov7-v6.2(2)/VOCdevkit/")
if not os.path.isdir(data_base_dir):
    os.mkdir(data_base_dir)
work_sapce_dir = os.path.join(data_base_dir, "VOC/")
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)
annotation_dir = os.path.join(work_sapce_dir, "Annotations/")
if not os.path.isdir(annotation_dir):
    os.mkdir(annotation_dir)
clear_hidden_files(annotation_dir)
image_dir = os.path.join(work_sapce_dir, "JPEGImages/")
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
clear_hidden_files(image_dir)
yolo_labels_dir = os.path.join(work_sapce_dir, "YOLOLabels/")
if not os.path.isdir(yolo_labels_dir):
    os.mkdir(yolo_labels_dir)
clear_hidden_files(yolo_labels_dir)
yolov7_images_dir = os.path.join(data_base_dir, "images/")
if not os.path.isdir(yolov7_images_dir):
    os.mkdir(yolov7_images_dir)
clear_hidden_files(yolov7_images_dir)
yolov7_labels_dir = os.path.join(data_base_dir, "labels/")
if not os.path.isdir(yolov7_labels_dir):
    os.mkdir(yolov7_labels_dir)
clear_hidden_files(yolov7_labels_dir)
yolov7_images_train_dir = os.path.join(yolov7_images_dir, "train/")
if not os.path.isdir(yolov7_images_train_dir):
    os.mkdir(yolov7_images_train_dir)
clear_hidden_files(yolov7_images_train_dir)
yolov7_images_test_dir = os.path.join(yolov7_images_dir, "val/")
if not os.path.isdir(yolov7_images_test_dir):
    os.mkdir(yolov7_images_test_dir)
clear_hidden_files(yolov7_images_test_dir)
yolov7_labels_train_dir = os.path.join(yolov7_labels_dir, "train/")
if not os.path.isdir(yolov7_labels_train_dir):
    os.mkdir(yolov7_labels_train_dir)
clear_hidden_files(yolov7_labels_train_dir)
yolov7_labels_test_dir = os.path.join(yolov7_labels_dir, "val/")
if not os.path.isdir(yolov7_labels_test_dir):
    os.mkdir(yolov7_labels_test_dir)
clear_hidden_files(yolov7_labels_test_dir)

train_file = open(os.path.join(wd, "yolov7_train.txt"), 'w')
test_file = open(os.path.join(wd, "yolov7_val.txt"), 'w')
train_file.close()
test_file.close()
train_file = open(os.path.join(wd, "yolov7_train.txt"), 'a')
test_file = open(os.path.join(wd, "yolov7_val.txt"), 'a')
list_imgs = os.listdir(image_dir)  # list image files
prob = random.randint(1, 100)
print("Probability: %d" % prob)
for i in range(0, len(list_imgs)):
    path = os.path.join(image_dir, list_imgs[i])
    if os.path.isfile(path):
        image_path = image_dir + list_imgs[i]
        voc_path = list_imgs[i]
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
        (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)
        label_name = nameWithoutExtention + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_name)
    prob = random.randint(1, 100)
    print("Probability: %d" % prob)
    if (prob < TRAIN_RATIO):  # train dataset
        if os.path.exists(annotation_path):
            train_file.write(image_path + '\n')
            convert_label(nameWithoutExtention)  # convert label
            copyfile(image_path, yolov7_images_train_dir + voc_path)
            copyfile(label_path, yolov7_labels_train_dir + label_name)
    else:  # test dataset
        if os.path.exists(annotation_path):
            test_file.write(image_path + '\n')
            convert_label(nameWithoutExtention)  # convert label
            copyfile(image_path, yolov7_images_test_dir + voc_path)
            copyfile(label_path, yolov7_labels_test_dir + label_name)
train_file.close()
test_file.close()





