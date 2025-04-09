import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
file_path = r"C:\Users\Lhtooo\Desktop\raw_yy_250326201148_0326201652.csv"
data = pd.read_csv(file_path)

# 提取左右眼和双眼的注视点
left_eye_x = data['Gaze Point Left X[px]']
left_eye_y = data['Gaze Point Left Y[px]']
right_eye_x = data['Gaze Point Right X[px]']
right_eye_y = data['Gaze Point Right Y[px]']
bino_eye_x = data['Gaze Point X[px]']
bino_eye_y = data['Gaze Point Y[px]']
trigger = data['Studio Event Index']

# 设定距离阈值为 150 像素
distance_threshold = 150


# 计算双眼注视点一致性（欧几里得距离）
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 找到所有唯一且有效的 Studio Event Index 值，并排序
unique_stages = np.sort(data['Studio Event Index'].dropna().unique())

# 找到各阶段起始索引
stage_starts = []
for stage in unique_stages:
    start_index = data[data['Studio Event Index'] == stage].index.min()
    if start_index is not None:
        stage_starts.append(start_index)

# 确保包含最后一个索引
stage_starts.sort()
stage_starts.append(len(data))

# 用于存储每个阶段的分析结果
stage_data = []

# 用于计算异常帧数
consecutive_abnormal_frames = 0
max_consecutive_abnormal_frames = 0
total_abnormal_frames = 0

# 为每个阶段进行分析
for i in range(len(stage_starts) - 1):
    start_idx = stage_starts[i]
    end_idx = stage_starts[i + 1]
    stage_segment = data.loc[start_idx:end_idx - 1]

    # 提取当前阶段的注视点
    left_eye_x_stage = stage_segment['Gaze Point Left X[px]']
    left_eye_y_stage = stage_segment['Gaze Point Left Y[px]']
    right_eye_x_stage = stage_segment['Gaze Point Right X[px]']
    right_eye_y_stage = stage_segment['Gaze Point Right Y[px]']

    # 计算双眼注视点一致性（欧几里得距离）
    distances_stage = euclidean_distance(left_eye_x_stage, left_eye_y_stage, right_eye_x_stage, right_eye_y_stage)
    abnormal_gaze_points_stage = distances_stage > distance_threshold

    # 计算左右眼注视点稳定性（标准差）
    std_left_x = np.std(left_eye_x_stage)
    std_left_y = np.std(left_eye_y_stage)
    std_right_x = np.std(right_eye_x_stage)
    std_right_y = np.std(right_eye_y_stage)

    # 生成每个阶段的注视点分析结果
    stage_data.append({
        'Stage Index': i + 1,
        'Left Eye Std X': std_left_x,
        'Left Eye Std Y': std_left_y,
        'Right Eye Std X': std_right_x,
        'Right Eye Std Y': std_right_y,
    })

    # 统计连续异常帧数
    abnormal_gaze_points_arr = abnormal_gaze_points_stage.values if isinstance(abnormal_gaze_points_stage,
                                                                               pd.Series) else abnormal_gaze_points_stage
    for is_abnormal in abnormal_gaze_points_arr:
        if is_abnormal:
            consecutive_abnormal_frames += 1
        else:
            max_consecutive_abnormal_frames = max(max_consecutive_abnormal_frames, consecutive_abnormal_frames)
            consecutive_abnormal_frames = 0

    # 累计异常帧数
    total_abnormal_frames += np.sum(abnormal_gaze_points_arr)

max_consecutive_abnormal_frames = max(max_consecutive_abnormal_frames, consecutive_abnormal_frames)

# 打印总体的结果
print(f"最长连续异常帧数：{max_consecutive_abnormal_frames}")

# 将每个阶段的分析结果转换为 DataFrame
stage_df = pd.DataFrame(stage_data)

# 输出数据
print(f"最大连续异常帧数：{max_consecutive_abnormal_frames}")
print(stage_df)

# 可视化每个阶段的注视点稳定性柱状图（左眼与右眼）
plt.figure(figsize=(12, 8))

# 设定条形图宽度和间隔
bar_width = 0.35
index = np.arange(len(stage_data))

# 绘制左眼和右眼的标准差柱状图
plt.bar(index, stage_df['Left Eye Std X'], bar_width, color='pink', label='Left Eye X Std')
plt.bar(index + bar_width, stage_df['Right Eye Std X'], bar_width, color='green', label='Right Eye X Std')

# 添加标题和标签
plt.title("Stability of Eye Gaze Points per Stage (Left vs Right Eye)", fontsize=16)
plt.xlabel("Stage Index", fontsize=12)
plt.ylabel("Standard Deviation (X)", fontsize=12)
plt.xticks(index + bar_width / 2, [f"Stage {i + 1}" for i in range(len(stage_data))])
plt.legend()

# 美化图表：设置网格、背景和文本大小
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 展示图形
plt.show()
