import os
import cv2
import socket
import numpy as np
import time
import csv
from datetime import datetime
import pytz

# 创建文件夹（如果不存在的话）
def create_folder(folder_path):
    if not os.path.exists(folder_path):  # 如果文件夹不存在
        os.makedirs(folder_path)  # 创建文件夹
        print(f"文件夹 {folder_path} 已创建")
    else:
        print(f"文件夹 {folder_path} 已存在")

def capture_and_save_video_from_udp(udp_address, save_folder, output_filename):
    # 创建文件夹（如果不存在的话）
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 打开视频流（UDP）
    video_capture = cv2.VideoCapture(udp_address)

    if not video_capture.isOpened():
        print(f"无法打开视频流 {udp_address}")
        return

    # 获取视频流的帧率（fps）和帧的大小
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    fps = 75
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 使用 VideoWriter 创建一个视频文件
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码
    output_video_path = os.path.join(save_folder, output_filename)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 创建CSV文件并写入表头
    csv_filename = os.path.join(save_folder, 'frame_times.csv')
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame Number', 'System Time (ns)', 'real time'])

    frame_number = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("无法捕获视频帧，检查视频流是否正常")
            break

        # 获取当前系统时间，精确到纳秒
        current_time_ns = time.time_ns()  # 返回纳秒级时间戳

        timestamp_sec = int(current_time_ns / 1e9)
        # 获取 UTC 时区的时间 
        utc_time = datetime.fromtimestamp(timestamp_sec, pytz.utc) 
        # 转换为北京时间（UTC+8） 
        beijing_tz = pytz.timezone("Asia/Shanghai") 
        beijing_time = utc_time.astimezone(beijing_tz) 
        # 获取纳秒部分 
        nanoseconds = current_time_ns % 1_000_000_000 
        # 格式化输出北京时间，包括纳秒部分 
        formatted_time = beijing_time.strftime("%Y-%m-%d %H:%M:%S") 
        formatted_time_with_ns = f"{formatted_time}.{nanoseconds:09d}"


        # 将帧编号和系统时间写入CSV文件
        with open(csv_filename, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([frame_number, current_time_ns, formatted_time_with_ns])

        # 将每一帧写入视频文件
        video_writer.write(frame)

        frame_number += 1

        # # 显示当前帧
        # cv2.imshow('Video Stream', frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        #     break

    # 释放视频捕获和写入对象
    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()


# # 示例用法
# if __name__ == "__main__":
    
#     save_folder = "data/video"  # 存储视频的文件夹
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     new_folder_path = os.path.join(save_folder, current_time)
#     create_folder(new_folder_path)
#     print(new_folder_path)
    
#     # UDP 地址格式：udp://<IP>:<端口号>
#     udp_address = "udp://127.0.0.1:8848"  # 这里替换为实际的UDP流地址
#     output_filename="video.avi"
#     capture_and_save_video_from_udp(udp_address, save_folder, output_filename)

def get_video(save_folder):
    # UDP 地址格式：udp://<IP>:<端口号>
    udp_ip = '127.0.0.1'
    udp_port = 8848
    udp_address = "udp://127.0.0.1:8848"  # 这里替换为实际的UDP流地址
    output_filename="video.avi"
    capture_and_save_video_from_udp(udp_address, save_folder, output_filename)

