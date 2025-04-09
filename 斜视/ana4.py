import pandas as pd
import numpy as np

# 加载数据
file_path = r"C:\Users\Lhtooo\Desktop\raw_yy_250326201148_0326201652.csv"
data = pd.read_csv(file_path)

# 确保数据按照时间戳排序，确保触发事件的顺序
data = data.sort_values(by='Recording Time Stamp[ms]')

# 创建一个空的列表，用于存储图像ID的列表
image_ids = []

# 遍历每行数据，根据trigger=202来为每一幅图像分配ID
current_image_id = None

prev_studio_event_index = None

for i, row in data.iterrows():
    studio_event_index = row['Studio Event Index']
    if pd.notna(studio_event_index):
        if prev_studio_event_index is None or studio_event_index > prev_studio_event_index:
            # 当 Studio Event Index 有变化时，更新当前图像标识
            current_image_id = row['Recording Time Stamp[ms]']  # 使用时间戳作为每个图像的标识
        prev_studio_event_index = studio_event_index
    image_ids.append(current_image_id)

# 将图像ID添加到数据中
data['image_id'] = image_ids

# 计算每一帧的眼动距离
data['eye_distance'] = np.sqrt(
    (data['Gaze Point Left X[px]'] - data['Gaze Point Right X[px]'])**2 +
    (data['Gaze Point Left Y[px]'] - data['Gaze Point Right Y[px]'])**2
)

# 按图像ID进行分组，并计算每组的最大值、最小值和阈值
eye_distance_stats = data.groupby('image_id')['eye_distance'].agg(['max', 'min'])

# 计算每幅图像的阈值（最大值 - 最小值）
eye_distance_stats['threshold'] = eye_distance_stats['max'] - eye_distance_stats['min']

# 打印输出每幅图像的统计结果
print(eye_distance_stats)
