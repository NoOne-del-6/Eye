import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载CSV文件
file_path = r"C:\Users\Lhtooo\Desktop\raw_yy_250326201148_0326201652.csv"
data = pd.read_csv(file_path)

# 计算左右眼视线位置的欧几里得距离
x_diff = data['Gaze Point Left X[px]'] - data['Gaze Point Right X[px]']
y_diff = data['Gaze Point Left Y[px]'] - data['Gaze Point Right Y[px]']
euclidean_distance = np.sqrt(x_diff**2 + y_diff**2)

# 定义异常视线点的阈值（150像素）
threshold = 150

# 创建图表
plt.figure(figsize=(10, 5))
plt.plot(euclidean_distance, color='lightblue', label="Euclidean Distance")
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold} pixels)')

# 给图表添加标题和标签
abnormal_ratio = np.sum(euclidean_distance > threshold) / len(euclidean_distance)
plt.title(f'Euclidean Distance Between Eye Gaze Points\nAbnormal Gaze Points Ratio: {abnormal_ratio:.4f} ({np.sum(euclidean_distance > threshold)}/{len(euclidean_distance)})')
plt.xlabel('Frame Index')
plt.ylabel('Distance (pixels)')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()
