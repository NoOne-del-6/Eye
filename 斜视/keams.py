import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取数据
file_paths = [
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/stra.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi1.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi2.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi3.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi4.csv',
    'C:/Users/DeepGaze/Desktop/example/picture_free_viewing/data/xieshi5.csv'
]

# 读取每个文件的数据
data_files = [pd.read_csv(file_path) for file_path in file_paths]

# 计算每个人每幅图像的阈值
def calculate_threshold_for_images(data):
    # 创建一个空的列表，用于存储图像ID
    image_ids = []
    
    # 用于标记每幅图像的ID
    current_image_id = None
    
    # 遍历数据，根据trigger=202来为每一幅图像分配ID
    for i, row in data.iterrows():
        if row['trigger'] == 202:  # 新的一幅图像
            if current_image_id is None or (current_image_id is not None and row['trigger'] == 202):
                current_image_id = row['timestamp']  # 使用时间戳作为每个图像的标识
        image_ids.append(current_image_id)
    
    # 将图像ID添加到数据中
    data['image_id'] = image_ids

    # 计算每一帧的眼动距离
    data['eye_distance'] = np.sqrt(
        (data['left_eye_gaze_position_x'] - data['right_eye_gaze_position_x'])**2 +
        (data['left_eye_gaze_position_y'] - data['right_eye_gaze_position_y'])**2
    )

    # 按图像ID进行分组，并计算每组的最大值、最小值
    eye_distance_stats = data.groupby('image_id')['eye_distance'].agg(['max', 'min'])

    # 计算阈值（最大值 - 最小值）
    eye_distance_stats['threshold'] = eye_distance_stats['max'] - eye_distance_stats['min']

    # 返回所有图像的阈值
    return eye_distance_stats['threshold'].values

# 存储每个人每幅图像的阈值
thresholds_per_image = {i: [] for i in range(9)}  # 9幅图像

# 对每个文件的数据进行阈值计算
for idx, data in enumerate(data_files):
    thresholds = calculate_threshold_for_images(data)
    for i in range(len(thresholds)):
        thresholds_per_image[i].append(thresholds[i])

# 聚类每一幅图像的阈值，生成聚类结果
image_cluster_results = []

for image_id, thresholds in thresholds_per_image.items():
    thresholds = np.array(thresholds).reshape(-1, 1)  # 聚类需要的格式
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(thresholds)
    
    # 存储聚类结果
    result = {
        'Image': f'Image {image_id + 1}',
        'Thresholds': thresholds.flatten(),
        'Clusters': clusters
    }
    image_cluster_results.append(result)

# 展示聚类结果
for result in image_cluster_results:
    print(f"Results for {result['Image']}:")
    print(f"Thresholds: {result['Thresholds']}")
    print(f"Clusters: {result['Clusters']}")
    print("")

# 可视化每幅图像的聚类结果
plt.figure(figsize=(10, 6))

# 为每幅图像生成可视化
for image_id, result in enumerate(image_cluster_results):
    plt.subplot(3, 3, image_id + 1)
    scatter = plt.scatter(range(len(result['Thresholds'])), result['Thresholds'], c=result['Clusters'], cmap='plasma', s=100)
    plt.title(f"{result['Image']}")
    plt.xlabel('Person')
    plt.ylabel('Threshold')
    plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()
