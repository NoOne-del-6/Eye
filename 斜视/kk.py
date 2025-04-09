import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 计算左右眼距离的函数
def calculate_eye_distance(df):
    left_eye_x = df['left_eye_gaze_position_x']
    left_eye_y = df['left_eye_gaze_position_y']
    right_eye_x = df['right_eye_gaze_position_x']
    right_eye_y = df['right_eye_gaze_position_y']
    distance = np.sqrt((left_eye_x - right_eye_x) ** 2 + (left_eye_y - right_eye_y) ** 2)
    return distance

# 处理每个人的数据
def process_data_for_person(df):
    df['eye_distance'] = calculate_eye_distance(df)
    triggers = df[df['trigger'] == 202].index
    image_data = []
    
    for i in range(len(triggers) - 1):
        start_idx = triggers[i] + 1
        end_idx = triggers[i + 1] if i + 1 < len(triggers) else len(df)
        
        image_segment = df[start_idx:end_idx]
        
        max_distance = image_segment['eye_distance'].max()
        min_distance = image_segment['eye_distance'].min()
        
        threshold = max_distance - min_distance
        image_data.append({'image': i + 1, 'max_distance': max_distance, 'min_distance': min_distance, 'threshold': threshold})
    
    return image_data

# 加载数据文件
file_paths = [
     'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/stra.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi1.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi2.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi3.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi4.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi5.csv'
]

data_files = [pd.read_csv(file_path) for file_path in file_paths]

# 处理每个人的数据并提取阈值
thresholds_per_person = []
for df in data_files:
    thresholds_per_person.append(process_data_for_person(df))

# 提取每个参与者的阈值数据
thresholds_array = []
for person_data in thresholds_per_person:
    thresholds_array.append([image['threshold'] for image in person_data])

thresholds_array = np.array(thresholds_array)

# 使用 StandardScaler 标准化数据
scaler = StandardScaler()
thresholds_array_scaled = scaler.fit_transform(thresholds_array)

# 使用肘部法则确定最优的聚类数目
sse = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(thresholds_array_scaled)
    sse.append(kmeans.inertia_)

# 绘制肘部法则的图形
plt.figure(figsize=(8, 6))
plt.plot(range(1, 8), sse, marker='o')
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
