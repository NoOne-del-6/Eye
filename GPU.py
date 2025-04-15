import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

# 设置 matplotlib 支持中文
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 环境变量设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# -------------------- 人脸识别函数 --------------------
def recognize_faces(frame: np.ndarray, device: str) -> list[np.array]:
    """
    Detects faces in the given image and returns the cropped facial regions.
    """
    def detect_face(frame: np.ndarray):
        mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
        if probs is None or probs[0] is None:
            return []
        bounding_boxes = bounding_boxes[probs > 0.9]
        return bounding_boxes

    bounding_boxes = detect_face(frame)
    facial_images = []
    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        facial_images.append(frame[y1:y2, x1:x2, :])
    return facial_images


# -------------------- 情绪分类 --------------------
def classify_emotion(emotion: str) -> str:
    """
    Classify the emotion into Positive, Negative, or Neutral categories.
    """
    positive_emotions = ["Happiness", "Surprise"]
    negative_emotions = ["Anger", "Contempt", "Disgust", "Fear", "Sadness"]
    
    if emotion in positive_emotions:
        return "Positive"
    elif emotion in negative_emotions:
        return "Negative"
    else:
        return "Neutral"


# -------------------- 情绪分析处理 --------------------
def process_video(video_path: str, fer, device: str, fps: float) -> list:
    """
    Process the video and return a list of frame data including timestamps and emotion scores.
    """
    cap = cv2.VideoCapture(video_path)
    all_scores = None
    frame_data = []  # 用于保存每帧得分 + 时间戳
    i = 0  # 帧计数器

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        if i % 65 != 0:  # 每隔一定帧数处理一次
            i += 1
            continue

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

        # 保存时间戳 + 得分到列表
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


# -------------------- 汇总分析 --------------------
def summarize_emotion_distribution(frame_data: list, fer) -> dict:
    """
    Summarize emotion distribution from the frame data.
    """
    # 统计情绪分布
    emotion_count = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for frame in frame_data:
        scores = [frame[emotion] for emotion in frame if emotion != "timestamp"]
        emotion_idx = np.argmax(scores)
        predicted_emotion = fer.idx_to_emotion_class[emotion_idx]
        emotion_type = classify_emotion(predicted_emotion)
        emotion_count[emotion_type] += 1

    highest_emotion = max(emotion_count, key=emotion_count.get)
    total = sum(emotion_count.values())
    proportions = {k: v / total for k, v in emotion_count.items()}

    return emotion_count, highest_emotion, proportions


# -------------------- 绘制柱状图 --------------------
def plot_emotion_distribution(emotion_count: dict, video_file: str,folder_path):
    """
    Plot the bar chart of emotion distribution and save it to the 'results' folder within the video file's directory.
    """
    labels = list(emotion_count.keys())
    sizes = list(emotion_count.values())

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, sizes, color=['lightgreen', 'lightcoral', 'lightblue'])

    ax.set_ylabel('频率')
    ax.set_xlabel('情绪类型')
    ax.set_title(f'{video_file} - 情绪类型分布')

    # 在柱状图上显示每个柱子的数值
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    # 获取视频文件所在的文件夹路径并创建 results 子文件夹
    results_folder = os.path.join(folder_path, 'results')
    os.makedirs(results_folder, exist_ok=True)

    # 设置保存柱状图的路径
    bar_chart_path = os.path.join(results_folder, f"{os.path.splitext(video_file)[0]}_emotion_bar_chart.png")

    # 保存柱状图
    plt.savefig(bar_chart_path)
    plt.close()

    return bar_chart_path


# -------------------- 主流程 --------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = get_model_list()[0]
    fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)

    folder_path = r"C:\Users\Lhtooo\Desktop\video"
    summary = []

    for video_file in os.listdir(folder_path):
        if video_file.endswith(".avi"):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing video: {video_path}")

            # 获取视频帧率
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # 处理视频
            frame_data, all_scores = process_video(video_path, fer, device, fps)

            # 汇总情绪分布
            emotion_count, highest_emotion, proportions = summarize_emotion_distribution(frame_data, fer)

            # 打印情绪分类结果
            print(f"Highest Emotion: {highest_emotion} "
                  f"(Positive: {proportions['Positive']:.2f}, "
                  f"Neutral: {proportions['Neutral']:.2f}, "
                  f"Negative: {proportions['Negative']:.2f})")

            # 绘制柱状图并保存到 'results' 文件夹
            bar_chart_path = plot_emotion_distribution(emotion_count, video_file,folder_path)

            # 保存情绪分析结果为 CSV 文件
            csv_file_path = os.path.join(os.path.dirname(video_path), 'results', f"{os.path.splitext(video_file)[0]}_emotion_scores.csv")
            df = pd.DataFrame(frame_data)
            df.to_csv(csv_file_path, index=False)

            # 写入总结信息
            summary.append(f"Video: {video_file}")
            summary.append(f"          Highest Emotion: {highest_emotion} \n\n"
                        #    f"(Positive: {proportions['Positive']:.2f}, "
                        #    f"Neutral: {proportions['Neutral']:.2f}, "
                        #    f"Negative: {proportions['Negative']:.2f})\n"
                           )


    # 将总结保存到 'results' 文件夹中
    summary_path = os.path.join(folder_path, 'results', "emotion_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.writelines(summary)

    print(f"所有视频的情绪分析结果已总结并保存为 {summary_path}")


if __name__ == "__main__":
    main()
