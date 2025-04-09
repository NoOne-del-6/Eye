import cv2
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace
import os


def analyze_video_emotions(video_path, csv_path):
    # 初始化情绪计数器
    emotion_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

    # 读取视频文件
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 确保 CSV 中的帧序号与视频帧一一对应
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(df) != frame_count:
        print("Warning: CSV frame count does not match video frame count!")

    # 对每一帧进行分析
    frame_emotions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 分析当前帧的情绪
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # 结果可能是一个包含字典的列表，我们只取第一个字典
        if isinstance(result, list):
            result = result[0]

        # 获取情绪
        emotion = result['dominant_emotion']

        # 分类为积极、中性或消极
        if emotion == 'happy':
            sentiment = 'Positive'
        elif emotion == 'neutral' or emotion == 'Surprise':
            sentiment = 'Neutral'
        else:
            sentiment = 'Negative'

        # 更新情绪计数器
        emotion_counts[sentiment] += 1
        frame_emotions.append(sentiment)

        # 更新CSV文件中的每帧情绪信息
        df.at[frame_number, 'Emotion'] = sentiment
        frame_number += 1

    # 保存更新后的 CSV 文件
    updated_csv_path = 'updated_' + os.path.basename(csv_path)
    df.to_csv(updated_csv_path, index=False)

    # 计算情绪比例
    total_frames = len(frame_emotions)
    positive_ratio = emotion_counts['Positive'] / total_frames
    neutral_ratio = emotion_counts['Neutral'] / total_frames
    negative_ratio = emotion_counts['Negative'] / total_frames

    # 绘制情绪柱状图
    emotions = ['Positive', 'Neutral', 'Negative']
    ratios = [positive_ratio, neutral_ratio, negative_ratio]

    plt.bar(emotions, ratios)
    plt.xlabel('Emotion')
    plt.ylabel('Proportion')
    plt.title('Emotion Proportions in Video')
    plt.savefig('emotion_proportions.png')  # 保存柱状图
    plt.show()

    print(
        f"Emotion analysis complete. Updated CSV saved as '{updated_csv_path}' and chart saved as 'emotion_proportions.png'.")


# 使用示例：
video_path = r"E:\video.avi"  # 替换为视频文件路径
csv_path = r"E:\frame_times.csv"  # 替换为CSV文件路径

analyze_video_emotions(video_path, csv_path)
