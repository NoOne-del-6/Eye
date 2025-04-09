import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# 加载数据
file_path = r"C:\Users\Lhtooo\Desktop\生理信号分析\data\眼动\lht.csv"  # 替换为您的文件路径
data = pd.read_csv(file_path)

# 筛选有效数据（从第一个 trigger=1 开始）
data = data[data['trigger'].cumsum() > 0].reset_index(drop=True)

# 添加图片标识符
data['image_id'] = data['trigger'].cumsum()

# 参数
x_col = 'bino_eye_gaze_position_x'  # 二眼注视点x坐标
y_col = 'bino_eye_gaze_position_y'  # 二眼注视点y坐标
time_col = 'timestamp'  # 时间戳
distance_threshold = 30  # 视角像素移动阈值，判定注视
duration_threshold = 100  # 持续时间阈值，判定注视（ms）
sampling_rate = 200  # 每秒采样率

# 初始化结果
results = []

# 按图片分组
for image_id, group in data.groupby('image_id'):
    fixations = []  # 注视
    saccades = []  # 眼跳
    saccade_lengths = []  # 眼跳长度
    current_fixation = []  # 当前注视点集合
    start_time = group[time_col].iloc[0]  # 当前注视起始时间

    # 遍历每个时间点
    for i in range(1, len(group)):
        # 前一个点和当前点坐标
        prev_point = (group[x_col].iloc[i - 1], group[y_col].iloc[i - 1])
        curr_point = (group[x_col].iloc[i], group[y_col].iloc[i])

        # 计算欧几里得距离
        dist = euclidean(prev_point, curr_point)

        # 判断是否属于同一注视
        if dist < distance_threshold:
            current_fixation.append(curr_point)
        else:
            # 如果注视结束，记录注视
            if len(current_fixation) > 0:
                duration = len(current_fixation) * (1000 / sampling_rate)  # 注视时间
                if duration >= duration_threshold:  # 只有超过时间阈值的才算注视
                    fixations.append({
                        "duration": duration,
                        "start_time": start_time,
                        "end_time": group[time_col].iloc[i - 1],
                        "fixation_points": current_fixation
                    })

            # 更新为新注视点
            current_fixation = [curr_point]
            start_time = group[time_col].iloc[i]

            # 记录眼跳信息
            saccades.append(curr_point)
            saccade_lengths.append(dist)

    # 统计指标
    avg_fixation_count = len(fixations) / len(group[time_col].unique())  # 平均注视次数
    avg_fixation_duration = np.mean([f['duration'] for f in fixations]) if fixations else 0  # 平均注视时间
    avg_saccade_count = len(saccades) / len(group[time_col].unique())  # 平均眼跳次数
    avg_saccade_length = np.mean(saccade_lengths) if saccade_lengths else 0  # 平均眼跳路径长度

    # 保存当前图片的指标
    results.append({
        "image_id": image_id,
        "fixation_count": len(fixations),
        "avg_fixation_duration": avg_fixation_duration,
        "saccade_count": len(saccades),
        "avg_saccade_length": avg_saccade_length
    })

# 转为DataFrame
results_df = pd.DataFrame(results)

# 输出结果到控制台
print(results_df)

# 将结果保存为 CSV 文件
output_file = r"C:\Users\Lhtooo\Desktop\生理信号分析\output_eye_movement_analysis.csv"
results_df.to_csv(output_file, index=False)
# print(f"结果已保存到: {output_file}")
