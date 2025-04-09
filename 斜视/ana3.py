import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
file_path = r"C:\Users\Lhtooo\Desktop\raw_yy_250326201148_0326201652.csv"
data = pd.read_csv(file_path)
valid_data = data.dropna(subset=['Studio Event Index'])
specific_values = [1, 2, 3, 4, 5,6,7,8,9]
# 筛选出符合特定值的行
specific_data = valid_data[valid_data['Studio Event Index'].isin(specific_values)]
target_change_indices = specific_data.index
# 采样频率 200Hz，即每帧的时间间隔为 0.005 秒（5 毫秒）
sampling_interval = 0.006  # 每帧的时间间隔为 5 毫秒

# 存储每张图片跳转的反应时间
reaction_times_per_image = []

# 记录每张图片的反应时间
for i in range(1, len(target_change_indices)):
    # 获取当前目标变化的时间戳（目标变化时间）
    target_change_time = data['Recording Time Stamp[ms]'].iloc[target_change_indices[i]]
    
    # 假设时间单位为毫微秒（nanoseconds），转换为秒（除以 1e9）
    target_change_time = target_change_time / 1e3  # 转换为秒（如果单位是毫微秒）
    
    # 获取目标变化之前的注视点（注视点位置）
    previous_x = data['Gaze Point X[px]'].iloc[target_change_indices[i]]
    previous_y = data['Gaze Point Y[px]'].iloc[target_change_indices[i]]
    
    # 遍历数据，寻找眼动开始的时刻
    start_reaction_time = None
    for j in range(target_change_indices[i], len(data) - 1):
        current_x = data['Gaze Point X[px]'].iloc[j]
        current_y = data['Gaze Point Y[px]'].iloc[j]
        
        # 计算位移（欧几里得距离）
        displacement = np.sqrt((current_x - previous_x)**2 + (current_y - previous_y)**2)
        
        # 设置一个位移阈值，假设50像素为合适的值
        displacement_threshold = 80
        
        # 如果位移超过阈值，认为眼动开始
        if displacement > displacement_threshold:
            start_reaction_time = data['Recording Time Stamp[ms]'].iloc[j]
            start_reaction_time = start_reaction_time / 1e3  # 转换为秒
            break
        
        # 更新前一个位置
        previous_x, previous_y = current_x, current_y
    
    # 如果找到了眼动开始的时间，则计算反应时间
    if start_reaction_time:
        reaction_time = start_reaction_time - target_change_time
        # 增加最小反应时间阈值，确保反应时间为合理的正数（例如5毫秒，即0.005秒）
        if reaction_time > sampling_interval:  # 设置最小反应时间阈值为5毫秒
            reaction_times_per_image.append(reaction_time)

# 打印每张图片的反应时间
for idx, reaction_time in enumerate(reaction_times_per_image, 1):
    print(f"图片 {idx} 跳转的反应时间：{reaction_time:.4f} 秒")

# 如果需要计算平均反应时间
if reaction_times_per_image:
    average_reaction_time = np.mean(reaction_times_per_image)
    print(f"所有图片的平均反应时间：{average_reaction_time:.4f} 秒")
else:
    print("没有有效的反应时间数据")

# 可视化每张图片的反应时间
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(reaction_times_per_image) + 1), reaction_times_per_image, color='skyblue')

# 设置图表标题和标签
plt.title('Reaction Time per Image', fontsize=16)
plt.xlabel('Image Index', fontsize=12)
plt.ylabel('Reaction Time (seconds)', fontsize=12)
plt.xticks(range(1, len(reaction_times_per_image) + 1))  # 设置X轴标签为图片编号
plt.grid(True, linestyle='--', alpha=0.7)

# 显示柱状图
plt.show()
