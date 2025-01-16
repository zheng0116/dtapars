import cv2
import mediapipe as mp
import json
import base64
import os
from os.path import join, exists
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

""" Media Labeling with MediaPipe Face Detection"""


def setup_directories(base_dir):
    dirs = [join(base_dir, "JPEGImages"), join(base_dir, "Annotations")]
    for dir_path in dirs:
        if not exists(dir_path):
            os.makedirs(dir_path)
    return dirs[0], dirs[1]


def get_face_info(image, max_faces=8):
    """使用MediaPipe获取多个人脸的关键点和人脸框"""
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    try:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # 1. 先检测人脸框
        with mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1,
        ) as face_detection:
            detection_results = face_detection.process(image)

            if not detection_results or not detection_results.detections:
                return None, None, "No face detection results"

            # 获取前N个置信度最高的人脸（最多max_faces个）
            detections = sorted(
                detection_results.detections, key=lambda x: x.score[0], reverse=True
            )[:max_faces]

            face_results = []
            for detection in detections:
                # 获取人脸框坐标
                bbox = detection.location_data.relative_bounding_box
                x1 = max(0, bbox.xmin * width)
                y1 = max(0, bbox.ymin * height)
                w = min(bbox.width * width, width - x1)
                h = min(bbox.height * height, height - y1)

                # 2. 裁剪人脸区域（适当扩大范围以确保不会裁掉关键点）
                margin = 0.1  # 10%的边距
                roi_x1 = max(0, int(x1 - w * margin))
                roi_y1 = max(0, int(y1 - h * margin))
                roi_x2 = min(width, int(x1 + w * (1 + 2 * margin)))
                roi_y2 = min(height, int(y1 + h * (1 + 2 * margin)))
                face_roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

                # 3. 在人脸区域内检测关键点
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                    refine_landmarks=True,
                ) as face_mesh:
                    mesh_results = face_mesh.process(face_roi)

                    if not mesh_results.multi_face_landmarks:
                        continue

                    # 4. 调整关键点坐标（从ROI坐标转换回原图坐标）
                    landmarks = mesh_results.multi_face_landmarks[0]
                    for landmark in landmarks.landmark:
                        # 转换相对坐标为绝对坐标
                        landmark.x = (
                            landmark.x * (roi_x2 - roi_x1) / width + roi_x1 / width
                        )
                        landmark.y = (
                            landmark.y * (roi_y2 - roi_y1) / height + roi_y1 / height
                        )

                    face_results.append((landmarks, detection))

            if not face_results:
                return None, None, "No valid face landmarks detected"

            return face_results, None, None

    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"


def get_face_bbox(detection, img_width, img_height):
    """从FaceDetection结果获取人脸框"""
    try:
        bbox = detection.location_data.relative_bounding_box
        xmin = max(0, bbox.xmin * img_width)
        ymin = max(0, bbox.ymin * img_height)
        width = min(bbox.width * img_width, img_width - xmin)
        height = min(bbox.height * img_height, img_height - ymin)

        return [[xmin, ymin], [xmin + width, ymin + height]]
    except Exception as e:
        print(f"Error in get_face_bbox: {str(e)}")
        return None


def get_imageData(img, save_image_data=True):
    """将图像转换为base64编码"""
    if not save_image_data:
        return None
    try:
        image_data_binary = cv2.imencode(".jpg", img)[1].tobytes()
        return base64.b64encode(image_data_binary).decode()
    except Exception as e:
        print(f"Error in get_imageData: {str(e)}")
        return None


def AxisTransformation(w, h, landmarks, face_idx=0):
    """转换MediaPipe关键点到目标格式"""
    if not landmarks:
        return None, None, None

    # 定义关键点索引和对应的标签数字
    KEYPOINTS = {
        "left_eye": {
            "indexes": [
                33,  # 左眼左角
                160,  # 左眼左上点
                158,  # 左眼右上点
                133,  # 左眼右角
                153,  # 左眼右下点
                144,  # 左眼左下点
            ],
            "labels": list(range(0, 6)),  # 0-5
        },
        "right_eye": {
            "indexes": [
                362,  # 右眼左角
                385,  # 右眼左上点
                387,  # 右眼右上点
                263,  # 右眼右角
                373,  # 右眼右下点
                380,  # 右眼左下点
            ],
            "labels": list(range(6, 12)),  # 6-11
        },
        "nose": {"indexes": [1], "labels": [12]},
        "mouth": {
            "indexes": [
                61,  # 1
                185,  # 2
                40,  # 3
                39,  # 4
                37,  # 5
                0,  # 6
                267,  # 7
                269,  # 8
                270,  # 9
                409,  # 10
                291,  # 11
                375,  # 12
                321,  # 13
                405,  # 14
                314,  # 15
                17,  # 16
                84,  # 17
                181,  # 18
                91,  # 19
                146,  # 20
            ],
            "labels": list(range(13, 33)),  # 13-32
        },
    }

    try:
        points = []
        labels = []
        all_points = []

        for feature, data in KEYPOINTS.items():
            for idx, label_num in zip(data["indexes"], data["labels"]):
                point = landmarks.landmark[idx]
                point_coord = [point.x * w, point.y * h]
                points.append([point_coord])
                # 添加face_idx前缀到label
                labels.append(f"{face_idx}_{label_num}")
                all_points.append(point_coord)

        return points, labels, all_points
    except Exception as e:
        print(f"Error in AxisTransformation: {str(e)}")
        return None, None, None


def get_shapes_data(points, label, shape_type="point"):
    """生成标注数据"""
    info = {
        "label": f"{label}",
        "points": points,
        "group_id": None,
        "description": "",
        "shape_type": shape_type,
        "flags": {},
    }
    return info


def process_single_image(image_name, image_dir, annotation_dir, max_faces=8):
    """处理单张图片的函数，用于并行处理"""
    try:
        img_path = join(image_dir, image_name)
        src_img = cv2.imread(img_path)
        if src_img is None:
            return (image_name, "Failed to read image")

        face_results, _, error_msg = get_face_info(src_img, max_faces)
        if face_results is None:
            return (image_name, error_msg)

        Height, Width, _ = src_img.shape
        imageData = get_imageData(src_img)
        if imageData is None:
            return (image_name, "Failed to encode image data")

        json_base = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_name,
            "imageData": imageData,
            "imageHeight": Height,
            "imageWidth": Width,
        }

        # 处理每个检测到的人脸
        for face_idx, (facial_landmarks, face_detection) in enumerate(face_results):
            # 添加人脸框
            bbox_points = get_face_bbox(face_detection, Width, Height)
            if bbox_points:
                bbox_info = get_shapes_data(
                    bbox_points, f"face_{face_idx}", "rectangle"
                )
                json_base["shapes"].append(bbox_info)

            # 添加关键点
            points, labels, _ = AxisTransformation(
                Width, Height, facial_landmarks, face_idx
            )
            if not points or not labels:
                continue

            for point, label in zip(points, labels):
                shape_info = get_shapes_data(point, label, "point")
                json_base["shapes"].append(shape_info)

        # 保存JSON文件
        output_path = join(annotation_dir, f"{os.path.splitext(image_name)[0]}.json")
        with open(output_path, "w") as f:
            json.dump(json_base, f, indent=2)

        return (image_name, None)  # None 表示成功

    except Exception as e:
        return (image_name, f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    DELETE_FAILED_IMAGES = True

    # 设置进程数，默认使用CPU核心数
    NUM_PROCESSES = cpu_count()

    wd = os.getcwd()
    data_base_dir = join(wd, "helen")
    image_dir = join(data_base_dir, "JPEGImages")
    annotation_dir = join(data_base_dir, "Annotations")

    if not exists(data_base_dir):
        os.makedirs(data_base_dir)
    image_dir, annotation_dir = setup_directories(data_base_dir)

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    total_files = len(image_files)
    print(
        f"Starting processing of {total_files} images using {NUM_PROCESSES} processes..."
    )

    # 创建进程池并并行处理图片
    with Pool(NUM_PROCESSES) as pool:
        # 使用partial固定image_dir和annotation_dir参数
        process_func = partial(
            process_single_image,
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            max_faces=8,  # 设置最大处理人脸数
        )

        # 使用tqdm显示进度
        results = list(
            tqdm(
                pool.imap(process_func, image_files),
                total=total_files,
                desc="Processing images",
            )
        )

    # 处理结果统计
    failed_images = [(name, reason) for name, reason in results if reason is not None]
    processed = total_files - len(failed_images)

    print("\nProcessing complete:")
    print(f"Total images: {total_files}")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {len(failed_images)}")

    if failed_images:
        print("\nFailed images details:")
        for img_name, reason in failed_images:
            print(f"- {img_name}: {reason}")
            if DELETE_FAILED_IMAGES:
                try:
                    img_path = join(image_dir, img_name)
                    if exists(img_path):
                        os.remove(img_path)
                        print(f"Deleted: {img_name}")
                except Exception as e:
                    print(f"Failed to delete {img_name}: {str(e)}")

    if DELETE_FAILED_IMAGES:
        print(f"\nDeleted {len(failed_images)} failed images.")
