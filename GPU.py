import os
from typing import List
import numpy as np
import cv2
import torch
import pandas as pd
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from concurrent.futures import ThreadPoolExecutor  # 导入并行处理模块

# -------------------- 人脸识别函数 --------------------
def recognize_faces(frame: np.ndarray, device: str) -> List[np.array]:
    """
    通过 MTCNN 检测给定图像中的人脸，并返回裁剪后的人脸区域
    """
    def detect_face(frame: np.ndarray):
        mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)  # 检测人脸框
        if probs is None or probs[0] is None:
            return []
        bounding_boxes = bounding_boxes[probs > 0.9]  # 过滤掉低概率的人脸
        return bounding_boxes

    bounding_boxes = detect_face(frame)
    facial_images = []
    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]  # 获取人脸框的坐标
        facial_images.append(frame[y1:y2, x1:x2, :])  # 切割人脸区域
    return facial_images


# -------------------- 情绪分类函数 --------------------
def classify_emotion(emotion: str) -> str:
    """
    根据模型预测的情绪进行分类
    """
    positive_emotions = ["Happiness", "Surprise"]
    negative_emotions = ["Anger", "Contempt", "Disgust", "Fear", "Sadness"]
    neutral_emotions = ["Neutral"]

    if emotion in positive_emotions:
        return "Positive"
    elif emotion in negative_emotions:
        return "Negative"
    else:
        return "Neutral"


# -------------------- 视频处理函数 --------------------
def process_video_frame(video_path: str, device: str, fer: EmotiEffLibRecognizer) -> (List[dict], np.ndarray):
    """
    处理视频帧并返回每帧的情绪得分和其他相关数据
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    all_scores = None
    frame_data = []  # 用于保存每帧的得分和时间戳
    i = 0  # 帧计数器

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # BGR → RGB 转换
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        facial_images = recognize_faces(image_rgb, device)

        # 过滤空图
        facial_images = [img for img in facial_images if img is not None and img.size != 0]
        if len(facial_images) == 0:
            i += 1
            continue

        # 情绪识别
        emotions, scores = fer.predict_emotions(facial_images, logits=True)

        # 保存时间戳和得分
        timestamp = i / fps
        frame_data.append({
            "timestamp": timestamp,
            **{fer.idx_to_emotion_class[j]: scores[0][j] for j in range(scores.shape[1])}
        })

        # 累加全局得分
        if all_scores is not None:
            all_scores = np.concatenate((all_scores, scores))
        else:
            all_scores = scores

        i += 1

    cap.release()
    return frame_data, all_scores


# -------------------- 保存结果到文件 --------------------
def save_results_to_txt(all_results: dict, output_txt_path: str):
    """
    保存分析结果到一个txt文件中
    """
    with open(output_txt_path, "w",encoding="utf-8") as file:
        for video_file, result in all_results.items():
            file.write(f"Video File: {video_file}\n")
            file.write(f"Predicted Emotion: {result['predicted_emotion']}\n")
            file.write(f"Emotion Type: {result['emotion_type']}\n")
            file.write(f"CSV Path: {result['csv_path']}\n")
            file.write("\n")  # 每个视频结果之间空一行
    print(f"Results saved to {output_txt_path}")


# -------------------- 主流程 --------------------
def main():
    device = "cuda:0"  # 设置为 GPU 设备
    model_name = get_model_list()[0]
    
    # 视频文件夹路径
    input_folder = r"C:\Users\Lhtooo\Desktop\video"
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.avi')]  # 获取所有视频文件
    
    fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)  # 初始化情绪识别器
    
    # 创建一个字典来保存每个视频的结果
    all_results = {}

    # 使用线程池并行处理视频帧
    with ThreadPoolExecutor() as executor:
        futures = []
        for video_file in video_files:
            video_path = os.path.join(input_folder, video_file)
            futures.append(executor.submit(process_video_frame, video_path, device, fer))

        for future, video_file in zip(futures, video_files):
            frame_data, all_scores = future.result()

            # -------------------- 输出分析结果 --------------------
            score = np.mean(all_scores, axis=0)
            emotion_idx = np.argmax(score)
            predicted_emotion = fer.idx_to_emotion_class[emotion_idx]
            emotion_type = classify_emotion(predicted_emotion)
            print(f"Video: {video_file}, Predicted Emotion: {predicted_emotion}, Emotion Type: {emotion_type}")

            # 保存每帧情绪得分到 CSV
            df = pd.DataFrame(frame_data)
            video_csv_path = os.path.join(input_folder, f"{video_file}_frame_emotion_scores.csv")
            df.to_csv(video_csv_path, index=False)
            print(f"情绪得分已保存为：{video_csv_path}")

            # 保存每个视频的结果
            all_results[video_file] = {
                "predicted_emotion": predicted_emotion,
                "emotion_type": emotion_type,
                "csv_path": video_csv_path
            }

    # 结果保存为txt文件
    output_txt_path = os.path.join(input_folder, "analysis_results.txt")
    save_results_to_txt(all_results, output_txt_path)


if __name__ == "__main__":
    main()
