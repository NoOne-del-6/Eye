import pandas as pd
import matplotlib.pyplot as plt

# 加载单个人的数据文件
file_path_person1 = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\data\LHT.csv"
data_person1 = pd.read_csv(file_path_person1)

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

def preprocess_data(data, sampling_rate=200):
    # 添加 AOI 列
    data['aoi'] = data.apply(lambda row: get_aoi(row['bino_eye_gaze_position_x'], row['bino_eye_gaze_position_y']), axis=1)
    
    # 过滤无效的 AOI 数据
    data = data.dropna(subset=['aoi'])
    
    # 计算每个点对应的时间（秒）
    data['time_per_point'] = 1 / sampling_rate
    return data

# 对Person 1的数据进行预处理
data_person1 = preprocess_data(data_person1)

# 按 AOI 统计注视时间
time_per_aoi_person1 = data_person1.groupby('aoi')['time_per_point'].sum()

# 计算注视时间比例
time_ratio_person1 = time_per_aoi_person1 / time_per_aoi_person1.sum()

# 绘制柱状图
fig, ax = plt.subplots(figsize=(10, 6))
bars = time_ratio_person1.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black', linewidth=1.5)

# 添加数值标签
for bar in bars.patches:  # bars.patches 是柱状图条的列表
    height = bar.get_height()
    if height > 0:  # 仅在柱状条高度大于 0 时添加标签
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2%}', 
                ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

# 设置图形属性
plt.xlabel("AOI (Emotion)", fontsize=14)
plt.ylabel("Proportion of Total Viewing Time", fontsize=14)
plt.title("Viewing Time Proportion per AOI for Person 1", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
