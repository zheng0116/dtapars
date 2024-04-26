import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

"extract CCPD2020 plate dataset dataname to labels"
def allFilePath(rootPath, allFIleList):
    '''
    获取指定目录下所有以.jpg结尾的文件的路径，并将这些路径存储在一个列表中。
    '''
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            if temp.endswith(".jpg"):
                allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)

def order_points(pts):
    '''
    对给定的坐标点进行排序，使得列表中的第一个点是左上角，第二个点是右上角，第三个点是右下角，第四个点是左下角。返回排序后的坐标点列表。
    '''
    pts = pts[:4, :]
    rect = np.zeros((5, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def get_rect_and_landmarks(img_path):
    '''该函数用于从图像文件路径中解析出矩形框和关键点的坐标，并返回解析后的结果。'''
    file_name = img_path.split("/")[-1].split("-")
    landmarks_np = np.zeros((5, 2))
    rect = file_name[2].split("_")
    landmarks = file_name[3].split("_")
    rect_str = "&".join(rect)
    landmarks_str = "&".join(landmarks)
    rect = rect_str.split("&")
    landmarks = landmarks_str.split("&")
    rect = [int(x) for x in rect]
    landmarks = [int(x) for x in landmarks]
    for i in range(4):
        landmarks_np[i][0] = landmarks[2 * i]
        landmarks_np[i][1] = landmarks[2 * i + 1]
    landmarks_np_new = order_points(landmarks_np)
    return rect, landmarks, landmarks_np_new

def x1x2y1y2_yolo(rect, landmarks, img):
    h, w, c = img.shape
    rect[0] = max(0, rect[0])
    rect[1] = max(0, rect[1])
    rect[2] = min(w - 1, rect[2] - rect[0])
    rect[3] = min(h - 1, rect[3] - rect[1])
    annotation = np.zeros((1, 14))
    annotation[0, 0] = (rect[0] + rect[2] / 2) / w  # cx
    annotation[0, 1] = (rect[1] + rect[3] / 2) / h  # cy
    annotation[0, 2] = rect[2] / w  # w
    annotation[0, 3] = rect[3] / h  # h

    annotation[0, 4] = landmarks[0] / w  # l0_x
    annotation[0, 5] = landmarks[1] / h  # l0_y
    annotation[0, 6] = landmarks[2] / w  # l1_x
    annotation[0, 7] = landmarks[3] / h  # l1_y
    annotation[0, 8] = landmarks[4] / w  # l2_x
    annotation[0, 9] = landmarks[5] / h  # l2_y
    annotation[0, 10] = landmarks[6] / w  # l3_x
    annotation[0, 11] = landmarks[7] / h  # l3_y
    # annotation[0, 12] = landmarks[8] / w  # l4_x
    # annotation[0, 13] = landmarks[9] / h  # l4_y
    return annotation

def xywh2yolo(rect, landmarks_sort, img):
    h, w, c = img.shape
    rect[0] = max(0, rect[0])
    rect[1] = max(0, rect[1])
    rect[2] = min(w - 1, rect[2] - rect[0])
    rect[3] = min(h - 1, rect[3] - rect[1])
    annotation = np.zeros((1, 12))
    annotation[0, 0] = (rect[0] + rect[2] / 2) / w  # cx
    annotation[0, 1] = (rect[1] + rect[3] / 2) / h  # cy
    annotation[0, 2] = rect[2] / w  # w
    annotation[0, 3] = rect[3] / h  # h

    annotation[0, 4] = landmarks_sort[0][0] / w  # l0_x
    annotation[0, 5] = landmarks_sort[0][1] / h  # l0_y
    annotation[0, 6] = landmarks_sort[1][0] / w  # l1_x
    annotation[0, 7] = landmarks_sort[1][1] / h  # l1_y
    annotation[0, 8] = landmarks_sort[2][0] / w  # l2_x
    annotation[0, 9] = landmarks_sort[2][1] / h  # l2_y
    annotation[0, 10] = landmarks_sort[3][0] / w  # l3_x
    annotation[0, 11] = landmarks_sort[3][1] / h  # l3_y
    # annotation[0, 12] = landmarks_sort[4][0] / w  # l4_x
    # annotation[0, 13] = landmarks_sort[4][1] / h  # l4_y
    return annotation
def yolo2x1y1x2y2(annotation, img):
    h, w, c = img.shape
    rect = annotation[:, 0:4].squeeze().tolist()
    landmarks = annotation[:, 4:].squeeze().tolist()
    rect_w = w * rect[2]
    rect_h = h * rect[3]
    rect_x = int(rect[0] * w - rect_w / 2)
    rect_y = int(rect[1] * h - rect_h / 2)
    new_rect = [rect_x, rect_y, rect_x + rect_w, rect_y + rect_h]
    for i in range(5):
        landmarks[2 * i] = landmarks[2 * i] * w
        landmarks[2 * i + 1] = landmarks[2 * i + 1] * h
    return new_rect, landmarks

def update_txt(file_root = r"./dataset/JPEGImages/",save_img_path=r"./dataset/img/",save_txt_path="./dataset/Annotations/"):
    print(file_root, "start!!!!!") #file_root原始图片路径、save_img_path保存的图片路径、save_txt_path保存的label路径
    file_list = []
    count = 0
    allFilePath(file_root, file_list)
    # print(file_list)
    # exit()
    for img_path in file_list:
        count += 1
        # img_path = r"ccpd_yolo_test/02-90_85-173&466_452&541-452&553_176&556_178&463_454&460-0_0_6_26_15_26_32-68-53.jpg"
        text_path = img_path.replace(".jpg", ".txt")
        # 读取图片
        img = cv2.imread(img_path)
        rect, landmarks, landmarks_sort = get_rect_and_landmarks(img_path)
        # annotation=x1x2y1y2_yolo(rect,landmarks,img)
        annotation = xywh2yolo(rect, landmarks_sort, img)
        str_label = "0 "
        for i in range(len(annotation[0])):
            str_label = str_label + " " + str(annotation[0][i])
        str_label = str_label.replace('[', '').replace(']', '')
        str_label = str_label.replace(',', '') + '\n'
        # if os.path.exists(text_path):
        #     continue
        # else:
        shutil.move(img_path,os.path.join(os.path.join(save_img_path,os.path.basename(img_path))))
        text_path_save = os.path.join(save_txt_path,os.path.basename(text_path))

        # print(text_path_save)
        # exit()
        with open(text_path_save, "w") as f:
            f.write(str_label)

        print(text_path,"finished!")
        # print(count, img_path)
    print(os.getpid(),"end!!!")

def delete_non_jpg_images(image_folder):
    for filename in os.listdir(image_folder):
        if not filename.endswith(".jpg"):
            file_path = os.path.join(image_folder, filename)
            os.remove(file_path)
            print("删除完毕")

def move_files_to_folders(images_folder, folders_folder, labels_folder):
    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(images_folder, filename)
            label_path = os.path.join(images_folder, os.path.splitext(filename)[0] + ".txt")
            folder_path = os.path.join(folders_folder, filename)
            labels_folder_path = os.path.join(labels_folder, os.path.splitext(filename)[0] + ".txt")
            if not os.path.exists(folder_path) and not os.path.exists(labels_folder_path) and os.path.exists(label_path):
                # 不存在同名
                shutil.move(image_path, folder_path)
                shutil.move(label_path, labels_folder_path)


if __name__ == '__main__':
    # 1. 处理ccpd文件夹
    import multiprocessing
    pool = multiprocessing.Pool(processes=14)  # 这里使用4个进程
    files = []
    for dir in os.listdir(r"./dataset"):
        files.append(os.path.join(r"./dataset",dir))
    # 使用进程池执行任务
    results = pool.map(update_txt,files)
    # 关闭进程池，防止新任务被提交
    pool.close()
    # 等待所有任务完成
    pool.join()

