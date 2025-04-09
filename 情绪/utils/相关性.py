import pandas as pd
import numpy as np

# 读取数据文件
file_path = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\data\眼动\zzq.csv"
data_person = pd.read_csv(file_path)

# 找到非零触发器的索引
trigger_indices = data_person[data_person['trigger'] != 0].index

# 确保至少有两个非零触发器
if len(trigger_indices) < 2:
    raise ValueError("数据中找不到足够的非零触发器。")

# 获取第一个和第二个非零触发器之间的数据
start_index = trigger_indices[0]
end_index = trigger_indices[1]
data_interval = data_person.iloc[start_index:end_index + 1]
# 去除left_eye_valid和right_eye_valid为0的数据
data_interval = data_interval[(data_interval['left_eye_valid'] != 0)&(data_interval['right_eye_valid'] != 0)&(data_interval['bino_eye_valid'] != 0)]
# 定义感兴趣区域 (AOI) 的边界
aoi_bounds = {
    "anger": [0, 640, 0, 540],
    "disgust": [640, 1280, 0, 540],
    "fear": [1280, 1920, 0, 540],
    "joy": [0, 640, 540, 1080],
    "sadness": [640, 1280, 540, 1080],
    "surprise": [1280, 1920, 540, 1080]
}

# 创建空矩阵
emotion_matrix = np.zeros((len(data_interval), len(aoi_bounds)), dtype=int)

# 重新计算填充矩阵，使用局部索引
for local_index, (index, row) in enumerate(data_interval.iterrows()):
    for i, (emotion, bounds) in enumerate(aoi_bounds.items()):
        if bounds[0] <= row['bino_eye_gaze_position_x'] <= bounds[1] and bounds[2] <= row['bino_eye_gaze_position_y'] <= bounds[3]:
            emotion_matrix[local_index, i] = 1

# 输出矩阵
# print(emotion_matrix)


#灰度矩阵
huidu_matrix = [ 90.16880787,109.55032986 ,69.715761  ,106.70325231 , 42.46865451 ,105.63962384]
# 将情感矩阵与灰度矩阵相乘
result_matrix = np.dot(emotion_matrix, huidu_matrix)

# 输出结果矩阵
print(result_matrix)
