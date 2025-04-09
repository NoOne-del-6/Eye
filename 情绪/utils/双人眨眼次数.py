import pandas as pd
import matplotlib.pyplot as plt


# 加载两个人的数据文件
file_path_person1 = r"C:\Users\Lhtooo\Desktop\生理信号分析\data\眼动\tgm.csv"
file_path_person2 = r"C:\Users\Lhtooo\Desktop\生理信号分析\data\眼动\ydn.csv"
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
    # 添加 AOI 列
    data['aoi'] = data.apply(lambda row: get_aoi(row['bino_eye_gaze_position_x'], row['bino_eye_gaze_position_y']), axis=1)
    
    # 过滤无效的 AOI 数据
    data = data.dropna(subset=['aoi'])
    return data

# 判断眨眼
def is_blink(row):
    # 左右眼都无效视为眨眼
    if row['left_eye_valid'] == 0 and row['right_eye_valid'] == 0:
        return True
    # 左右眼瞳孔直径为 0 或者 NaN 时视为眨眼
    if row['left_eye_pupil_diameter_mm'] == 0 :
        return True
    if row['right_eye_pupil_diameter_mm'] == 0 :
        return True
    return False

# 对两个人的数据进行预处理
data_person1 = preprocess_data(data_person1)
data_person2 = preprocess_data(data_person2)

# 添加眨眼标识列
data_person1['is_blink'] = data_person1.apply(is_blink, axis=1)
data_person2['is_blink'] = data_person2.apply(is_blink, axis=1)

# 按 AOI 统计眨眼次数
blink_counts_person1 = data_person1[data_person1['is_blink']].groupby('aoi').size()
blink_counts_person2 = data_person2[data_person2['is_blink']].groupby('aoi').size()

# 合并结果，便于比较
comparison_df = pd.DataFrame({
    "Person 1": blink_counts_person1,
    "Person 2": blink_counts_person2
}).fillna(0)  # 如果某人某个 AOI 没有数据，用 0 填充

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 6))
bars = comparison_df.plot(kind='bar', ax=ax, color=['skyblue', 'orange'], edgecolor='black', linewidth=1.5)

# 添加数值标签
for bar_group in bars.containers:
    for bar in bar_group:
        height = bar.get_height()
        if height > 0:  # 仅在柱状条高度大于 0 时添加标签
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # 横向居中
                height + 0.5,  # 高度稍微上移一点
                f'{int(height)}',  # 格式化为整数
                ha='center', va='bottom', fontsize=10, color='black'
            )

# 设置图形属性
plt.xlabel("AOI (Emotion)", fontsize=14)
plt.ylabel("Blink Count", fontsize=14)
plt.title("Blink Counts per AOI by Person", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title="Person", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
