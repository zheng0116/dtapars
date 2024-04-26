import json
import os
import random
from os import listdir, getcwd
from os.path import join
from shutil import copyfile

classes = ['person']
TRAIN_RATIO = 60

wd = getcwd()
data_base_dir = join(wd, "dataset/")
annotation_dir = join(data_base_dir, "Annotations/")
image_dir = join(data_base_dir, "JPEGImages/")
yolov7_images_dir = join(data_base_dir, "images/")
yolov7_labels_dir = join(data_base_dir, "labels/")
yolov7_images_train_dir = join(yolov7_images_dir, "train/")
yolov7_images_val_dir = join(yolov7_images_dir, "val/")
yolov7_labels_train_dir = join(yolov7_labels_dir, "train/")
yolov7_labels_val_dir = join(yolov7_labels_dir, "val/")

for dir_path in [yolov7_images_dir, yolov7_labels_dir, yolov7_images_train_dir, 
                 yolov7_images_val_dir, yolov7_labels_train_dir, yolov7_labels_val_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def convert(json_file, out_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    img_width = data["imageWidth"]
    img_height = data["imageHeight"]
    bbox = data['shapes'][0]['points']
    keypoints = data['shapes'][1:]

    x_min, y_min = bbox[0]
    x_max, y_max = bbox[1]
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    output_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    for kp in keypoints:
        x_kp, y_kp = kp['points'][0]
        x_kp_norm = x_kp / img_width
        y_kp_norm = y_kp / img_height
        output_line += f" {x_kp_norm:.6f} {y_kp_norm:.6f} 2.000000"

    with open(out_file, 'w') as file:
        file.write(output_line + "\n")

image_files = listdir(image_dir)
random.shuffle(image_files)  

for image_file in image_files:
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        base_name = os.path.splitext(image_file)[0]
        json_file = join(annotation_dir, base_name + '.json')
        image_path = join(image_dir, image_file)

        # Decide whether it goes to train or val
        if random.randint(0, 100) < TRAIN_RATIO:
            target_image_dir = yolov7_images_train_dir
            target_label_dir = yolov7_labels_train_dir
        else:
            target_image_dir = yolov7_images_val_dir
            target_label_dir = yolov7_labels_val_dir

        convert(json_file, join(target_label_dir, base_name + '.txt'))
        copyfile(image_path, join(target_image_dir, image_file))

print("Dataset preparation is complete.")
