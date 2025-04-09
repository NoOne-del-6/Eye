import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import gaussian_kde

# 加载数据
file_path = r"C:\Users\Lhtooo\Desktop\raw_yy_250326201148_0326201652.csv"
data = pd.read_csv(file_path)

# 提取左右眼和双眼的注视点
left_eye_x = data['Gaze Point Left X[px]']
left_eye_y = data['Gaze Point Left Y[px]']
right_eye_x = data['Gaze Point Right X[px]']
right_eye_y = data['Gaze Point Right Y[px]']
bino_eye_x = data['Gaze Point X[px]']  # 双眼注视点X
bino_eye_y = data['Gaze Point Y[px]']  # 双眼注视点Y
trigger = data['Studio Event Index']

# 1. 计算双眼注视点一致性（欧几里得距离）
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# 设定距离阈值为150像素
distance_threshold = 150

# 计算每一帧的双眼注视点距离
distances = euclidean_distance(left_eye_x, left_eye_y, right_eye_x, right_eye_y)

# 标记双眼注视点异常的帧
abnormal_gaze_points = distances > distance_threshold

# 2. 计算注视点稳定性（标准差）
# 计算双眼注视点的标准差
std_bino_x = np.std(bino_eye_x)
std_bino_y = np.std(bino_eye_y)

# 3. 时间维度的注视行为
# 设定连续异常帧的阈值（例如超过100毫秒，即10帧，假设帧率为100Hz）
frame_rate = 100  # 假设的帧率，100帧/秒
frame_threshold = frame_rate * 0.1  # 对应100毫秒，10帧

# 统计连续异常帧数
consecutive_abnormal_frames = 0
max_consecutive_abnormal_frames = 0
for i in range(1, len(abnormal_gaze_points)):
    if abnormal_gaze_points[i]:
        consecutive_abnormal_frames += 1
    else:
        max_consecutive_abnormal_frames = max(max_consecutive_abnormal_frames, consecutive_abnormal_frames)
        consecutive_abnormal_frames = 0

max_consecutive_abnormal_frames = max(max_consecutive_abnormal_frames, consecutive_abnormal_frames)

# 打印总体的结果
print(f"总体注视点稳定性：标准差 (X): {std_bino_x:.2f}, 标准差 (Y): {std_bino_y:.2f}")
print(f"总体双眼注视点一致性：异常注视点数量（大于150像素距离）: {np.sum(abnormal_gaze_points)}")
print(f"总体时间维度的注视行为：最大连续异常帧数：{max_consecutive_abnormal_frames} 帧")

# 加载图像文件
image_path = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\images\xieshi.png"
image = Image.open(image_path)

# 获取图像尺寸
img_width, img_height = image.size

# 4. 创建热图：计算眼动点的密度分布
def generate_heatmap(x_data, y_data, width, height):
    # 创建一个二维高斯核密度估计
    xy = np.vstack([x_data, y_data])
    kde = gaussian_kde(xy)
    xmin, xmax = 0, width
    ymin, ymax = 0, height
    xgrid, ygrid = np.mgrid[xmin:xmax:width/100, ymin:ymax:height/100]
    grid_positions = np.vstack([xgrid.ravel(), ygrid.ravel()])
    z = kde(grid_positions).reshape(xgrid.shape)
    return xgrid, ygrid, z

# 生成左右眼的注视点热图
left_eye_x_data = left_eye_x[(left_eye_x > 0) & (left_eye_x < img_width) & (left_eye_y > 0) & (left_eye_y < img_height)]
left_eye_y_data = left_eye_y[(left_eye_x > 0) & (left_eye_x < img_width) & (left_eye_y > 0) & (left_eye_y < img_height)]

right_eye_x_data = right_eye_x[(right_eye_x > 0) & (right_eye_x < img_width) & (right_eye_y > 0) & (right_eye_y < img_height)]
right_eye_y_data = right_eye_y[(right_eye_x > 0) & (right_eye_x < img_width) & (right_eye_y > 0) & (right_eye_y < img_height)]

# 确保左右眼的注视点数据长度一致
min_len_left = min(len(left_eye_x_data), len(left_eye_y_data))
min_len_right = min(len(right_eye_x_data), len(right_eye_y_data))

# 如果有长度不一致的情况，取最小长度的数据
left_eye_x_data = left_eye_x_data[:min_len_left]
left_eye_y_data = left_eye_y_data[:min_len_left]

right_eye_x_data = right_eye_x_data[:min_len_right]
right_eye_y_data = right_eye_y_data[:min_len_right]

# 获取左右眼的热图数据
x_left, y_left, z_left = generate_heatmap(left_eye_x_data, left_eye_y_data, img_width, img_height)
x_right, y_right, z_right = generate_heatmap(right_eye_x_data, right_eye_y_data, img_width, img_height)

# 可视化左右眼注视点位置
plt.figure(figsize=(12, 8))
##plt.imshow(image, extent=[0, img_width, 0, img_height])
plt.scatter(left_eye_x, left_eye_y, s=1, alpha=0.5, label='Left Eye Gaze', color='blue')
plt.scatter(right_eye_x, right_eye_y, s=1, alpha=0.5, label='Right Eye Gaze', color='red')
plt.title("Eye Gaze Points and Image", fontsize=16)
plt.xlabel("X Position (pixels)", fontsize=12)
plt.ylabel("Y Position (pixels)", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)

# 展示图形
plt.show()

# 可视化热图
plt.figure(figsize=(12, 8))
plt.imshow(image, extent=[0, img_width, 0, img_height])  # 显示背景图像
plt.contourf(x_left, y_left, z_left, cmap="Blues", alpha=0.6)  # 左眼热图
plt.contourf(x_right, y_right, z_right, cmap="Reds", alpha=0.6)  # 右眼热图
plt.title("Heatmap of Eye Gaze Points", fontsize=16)
plt.xlabel("X Position (pixels)", fontsize=12)
plt.ylabel("Y Position (pixels)", fontsize=12)
plt.colorbar(label="Density")
plt.grid(True)

# 展示图形
plt.show()
