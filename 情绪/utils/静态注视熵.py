# import pandas as pd
# import numpy as np

# # 加载两个人的数据文件
# file_path_person1 = r"tgm.csv"  # 请替换为你的文件路径
# file_path_person2 = r"ydn.csv"   # 请替换为你的文件路径
# data_person1 = pd.read_csv(file_path_person1)
# data_person2 = pd.read_csv(file_path_person2)

# # 定义感兴趣区域 (AOI) 的边界
# aoi_bounds = {
#     "anger": [0, 640, 0, 540],        # Top-left
#     "disgust": [640, 1280, 0, 540],   # Top-center
#     "fear": [1280, 1920, 0, 540],     # Top-right
#     "joy": [0, 640, 540, 1080],       # Bottom-left
#     "sadness": [640, 1280, 540, 1080], # Bottom-center
#     "surprise": [1280, 1920, 540, 1080] # Bottom-right
# }

# # 添加 AOI 信息
# def get_aoi(x, y):
#     for emotion, bounds in aoi_bounds.items():
#         x_min, x_max, y_min, y_max = bounds
#         if x_min <= x <= x_max and y_min <= y <= y_max:
#             return emotion
#     return None

# def preprocess_data(data):
#     # 添加 AOI 列
#     data['aoi'] = data.apply(lambda row: get_aoi(row['bino_eye_gaze_position_x'], row['bino_eye_gaze_position_y']), axis=1)
    
#     # 过滤无效的 AOI 数据
#     data = data.dropna(subset=['aoi'])
#     return data

# # 对两个人的数据进行预处理
# data_person1 = preprocess_data(data_person1)
# data_person2 = preprocess_data(data_person2)

# # 统计注视次数
# def get_aoi_statistics(data):
#     aoi_counts = data['aoi'].value_counts()
#     total_counts = aoi_counts.sum()
    
#     # 计算每个AOI的注视次数百分比
#     aoi_percentages = (aoi_counts / total_counts) * 100
    
#     return aoi_counts, aoi_percentages

# # 统计每个人的AOI注视次数及百分比
# aoi_counts_person1, aoi_percentages_person1 = get_aoi_statistics(data_person1)
# aoi_counts_person2, aoi_percentages_person2 = get_aoi_statistics(data_person2)

# # 计算静态注视熵
# def calculate_entropy(percentages):
#     probabilities = percentages / 100  # 转换为概率
#     entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))  # 加上微小值避免log(0)
#     return entropy

# entropy_person1 = calculate_entropy(aoi_percentages_person1)
# entropy_person2 = calculate_entropy(aoi_percentages_person2)

# # 显示结果
# print("Person 1 AOI Counts:\n", aoi_counts_person1)
# print("Person 1 AOI Percentages:\n", aoi_percentages_person1)
# print("Person 1 Static Gaze Entropy:", entropy_person1)

# print("\nPerson 2 AOI Counts:\n", aoi_counts_person2)
# print("Person 2 AOI Percentages:\n", aoi_percentages_person2)
# print("Person 2 Static Gaze Entropy:", entropy_person2)
import pandas as pd
import numpy as np

# 加载两个人的数据文件
file_path_person1 = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\data\LHT.csv"  # 请替换为你的文件路径
file_path_person2 = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\data\LHT.csv"   # 请替换为你的文件路径
data_person1 = pd.read_csv(file_path_person1)
data_person2 = pd.read_csv(file_path_person2)

# 定义感兴趣区域 (AOI) 的边界
aoi_bounds = {
    "anger": [0, 640, 0, 540],        # Top-left
    "disgust": [640, 1280, 0, 540],   # Top-center
    "fear": [1280, 1920, 0, 540],     # Top-right
    "joy": [0, 640, 540, 1080],       # Bottom-left
    "sadness": [640, 1280, 540, 1080], # Bottom-center
    "surprise": [1280, 1920, 540, 1080] # Bottom-right
}

# 添加 AOI 信息
def get_aoi(x, y):
    for emotion, bounds in aoi_bounds.items():
        x_min, x_max, y_min, y_max = bounds
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return emotion
    return None

def preprocess_data(data):
    # 确保时间戳是整数格式
    data['timestamp'] = pd.to_numeric(data['timestamp'], errors='coerce')
    
    # 按时间排序
    data.sort_values(by='timestamp', inplace=True)
    
    # 计算每个注视的持续时间
    data['duration'] = data['timestamp'].diff().fillna(0)  # 计算时间间隔
    data['duration'] = data['duration'].abs()  # 确保持续时间为正数
    
    # 过滤出持续时间大于10ms的数据
    data = data[data['duration'] > 100]  # 过滤有效数据（10毫秒以上）

    # 添加 AOI 列
    data['aoi'] = data.apply(lambda row: get_aoi(row['bino_eye_gaze_position_x'], row['bino_eye_gaze_position_y']), axis=1)
    
    # 过滤无效的 AOI 数据
    data = data.dropna(subset=['aoi'])
    return data

# 对两个人的数据进行预处理
data_person1 = preprocess_data(data_person1)
data_person2 = preprocess_data(data_person2)

# 统计注视次数
def get_aoi_statistics(data):
    aoi_counts = data['aoi'].value_counts()
    total_counts = aoi_counts.sum()
    
    # 计算每个AOI的注视次数百分比
    aoi_percentages = (aoi_counts / total_counts) * 100
    
    return aoi_counts, aoi_percentages

# 统计每个人的AOI注视次数及百分比
aoi_counts_person1, aoi_percentages_person1 = get_aoi_statistics(data_person1)
aoi_counts_person2, aoi_percentages_person2 = get_aoi_statistics(data_person2)

# 计算静态注视熵
def calculate_entropy(percentages):
    probabilities = percentages / 100  # 转换为概率
    entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))  # 加上微小值避免log(0)
    return entropy

entropy_person1 = calculate_entropy(aoi_percentages_person1)
entropy_person2 = calculate_entropy(aoi_percentages_person2)

# 输出结果
print("Person 1 AOI Counts:\n", aoi_counts_person1)
print("Person 1 AOI Percentages:\n", aoi_percentages_person1)
print("Person 1 Entropy:", entropy_person1)

print("\nPerson 2 AOI Counts:\n", aoi_counts_person2)
print("Person 2 AOI Percentages:\n", aoi_percentages_person2)
print("Person 2 Entropy:", entropy_person2)
