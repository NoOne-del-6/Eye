import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import os
import time


def analyze_video_emotions_in_folder(folder_path):
    # 初始化情绪计数器
    emotion_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

    # 获取文件夹内所有视频文件
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4', '.mov'))]

    # 记录开始时间
    start_time = time.time()

    # 存储结果的txt文件路径
    txt_file_path = os.path.join(folder_path, "emotion_analysis_results.txt")

    # 打开txt文件进行写入
    with open(txt_file_path, "w",encoding="utf-8") as result_file:
        # 对每个视频进行情绪分析
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            print(f"Analyzing video: {video_path}")

            # 读取视频文件
            cap = cv2.VideoCapture(video_path)
            frame_number = 0

            # 获取视频的总帧数
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

                frame_number += 1

            cap.release()  # 释放视频文件资源

            # 计算情绪比例
            total_frames = len(frame_emotions)
            positive_ratio = emotion_counts['Positive'] / total_frames
            neutral_ratio = emotion_counts['Neutral'] / total_frames
            negative_ratio = emotion_counts['Negative'] / total_frames

            # 找出情绪占比最高的情绪
            emotion_ratios = {'Positive': positive_ratio, 'Neutral': neutral_ratio, 'Negative': negative_ratio}
            highest_emotion = max(emotion_ratios, key=emotion_ratios.get)

            # 将结果写入txt文件
            result_file.write(f"Video: {video_file}\n")
            result_file.write(
                f"Highest Emotion: {highest_emotion} (Positive: {positive_ratio:.2f}, Neutral: {neutral_ratio:.2f}, Negative: {negative_ratio:.2f})\n\n")

            # 绘制情绪柱状图
            emotions = ['Positive', 'Neutral', 'Negative']
            ratios = [positive_ratio, neutral_ratio, negative_ratio]

            # 确保保存路径是视频文件所在的文件夹
            save_path = os.path.dirname(video_path)
            plt.figure(figsize=(6, 4))
            plt.bar(emotions, ratios)
            plt.xlabel('Emotion')
            plt.ylabel('Proportion')
            plt.title(f'Emotion Proportions in Video: {video_file}')
            plt.savefig(os.path.join(save_path, f'{video_file}_emotion_proportions.png'))  # 保存到视频文件夹
            plt.close()

            print(f"Finished analyzing {video_file}. Emotion proportions chart saved.")

    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Emotion analysis complete for all videos. Total processing time: {total_time:.2f} seconds.")
    print(f"Emotion analysis results saved in: {txt_file_path}")

# 使用示例：
folder_path = r"C:\Users\Lhtooo\Desktop\data\情绪图片\ydn_female\20250411_160902" # 替换为包含视频的文件夹路径
analyze_video_emotions_in_folder(folder_path)