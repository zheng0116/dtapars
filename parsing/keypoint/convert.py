import json
import os
import random
from os import listdir, getcwd
from os.path import join
from shutil import copyfile
import re

# 配置参数
classes = ["face"]  # 改为face类别
TRAIN_RATIO = 80  # 训练集比例调整为80%
KEYPOINTS_PER_FACE = 33  # 每个人脸的关键点数量
MAX_FACES = 8  # 最大人脸数量

# 目录设置
wd = getcwd()
data_base_dir = join(wd, "dataset1/")
annotation_dir = join(data_base_dir, "Annotations/")
image_dir = join(data_base_dir, "JPEGImages/")
yolov11_images_dir = join(data_base_dir, "images/")
yolov11_labels_dir = join(data_base_dir, "labels/")
yolov11_images_train_dir = join(yolov11_images_dir, "train/")
yolov11_images_val_dir = join(yolov11_images_dir, "val/")
yolov11_labels_train_dir = join(yolov11_labels_dir, "train/")
yolov11_labels_val_dir = join(yolov11_labels_dir, "val/")

# 创建必要的目录
for dir_path in [
    yolov11_images_dir,
    yolov11_labels_dir,
    yolov11_images_train_dir,
    yolov11_images_val_dir,
    yolov11_labels_train_dir,
    yolov11_labels_val_dir,
]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def extract_face_index(label):
    """从标签中提取人脸索引，如 'face_0' -> 0, '0_12' -> 0"""
    try:
        if label.startswith("face_"):
            return int(label.split("_")[1])
        return int(label.split("_")[0])
    except:
        return None


def group_shapes_by_face(shapes):
    """将标注按人脸分组"""
    face_groups = {}  # 用于存储每个人脸的标注

    for shape in shapes:
        face_idx = extract_face_index(shape["label"])
        if face_idx is not None:
            if face_idx not in face_groups:
                face_groups[face_idx] = []
            face_groups[face_idx].append(shape)

    return face_groups


def convert_single_face(face_shapes, img_width, img_height, json_file, face_idx):
    """转换单个人脸的标注"""
    # 找到bbox和关键点
    bbox = None
    keypoints = []

    # 分离bbox和关键点
    for shape in face_shapes:
        if shape["shape_type"] == "rectangle" and shape["label"].startswith("face_"):
            bbox = shape
        elif shape["shape_type"] == "point":
            # 提取数字编号
            point_idx = int(shape["label"].split("_")[1])
            keypoints.append((point_idx, shape))

    # 调试信息
    if not bbox:
        print(f"Debug - {json_file} 人脸 {face_idx} 没有找到bbox")
        return None

    # 按关键点编号排序
    keypoints.sort(key=lambda x: x[0])

    # 处理bbox
    x_min, y_min = bbox["points"][0]
    x_max, y_max = bbox["points"][1]

    # 计算归一化的bbox坐标
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    # 构建输出
    output = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    # 添加关键点
    for _, kp in keypoints:
        x_kp, y_kp = kp["points"][0]
        x_kp_norm = x_kp / img_width
        y_kp_norm = y_kp / img_height
        output += f" {x_kp_norm:.6f} {y_kp_norm:.6f} 2.000000"

    return output


def convert(json_file, out_file):
    """转换标注文件，支持多人脸"""
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        img_width = data["imageWidth"]
        img_height = data["imageHeight"]

        # 按人脸分组标注
        face_groups = group_shapes_by_face(data["shapes"])

        if not face_groups:
            print(f"Debug - {json_file} 未找到任何人脸分组")
            return False

        # 输出调试信息
        for face_idx, shapes in face_groups.items():
            keypoint_count = sum(1 for s in shapes if s["shape_type"] == "point")
            bbox_count = sum(1 for s in shapes if s["shape_type"] == "rectangle")
            print(
                f"Debug - {json_file} 人脸 {face_idx}: {keypoint_count} 个关键点 + {bbox_count} 个bbox"
            )

        # 处理每个人脸，写入输出文件
        successful_converts = []
        for face_idx in sorted(face_groups.keys()):
            face_shapes = face_groups[face_idx]

            # 转换单个人脸
            result = convert_single_face(
                face_shapes, img_width, img_height, json_file, face_idx
            )
            if result:
                successful_converts.append(result)
            else:
                print(f"Debug - {json_file} 人脸 {face_idx} 转换失败")

        # 如果有成功转换的人脸，写入文件
        if successful_converts:
            with open(out_file, "w") as file:
                for result in successful_converts:
                    file.write(result + "\n")
            return True

        return False

    except Exception as e:
        print(f"处理 {json_file} 时出错: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    image_files = listdir(image_dir)
    random.shuffle(image_files)

    success_count = 0
    fail_count = 0

    for image_file in image_files:
        if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            base_name = os.path.splitext(image_file)[0]
            json_file = join(annotation_dir, base_name + ".json")
            image_path = join(image_dir, image_file)

            # 检查json文件是否存在
            if not os.path.exists(json_file):
                print(f"警告: 找不到对应的标注文件 {json_file}")
                fail_count += 1
                continue

            # 决定是训练集还是验证集
            if random.randint(1, 100) <= TRAIN_RATIO:
                target_image_dir = yolov11_images_train_dir
                target_label_dir = yolov11_labels_train_dir
            else:
                target_image_dir = yolov11_images_val_dir
                target_label_dir = yolov11_labels_val_dir

            # 转换和复制文件
            output_label_path = join(target_label_dir, base_name + ".txt")
            if convert(json_file, output_label_path):
                copyfile(image_path, join(target_image_dir, image_file))
                success_count += 1
            else:
                fail_count += 1
                # 如果转换失败，删除可能生成的空文件
                if os.path.exists(output_label_path):
                    os.remove(output_label_path)

    print(
        f"""
    数据集处理完成:
    - 成功处理: {success_count} 个文件
    - 处理失败: {fail_count} 个文件
    - 总计文件: {success_count + fail_count} 个
    """
    )
