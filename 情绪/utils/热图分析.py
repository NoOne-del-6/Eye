import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载CSV文件
file_path = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\data\眼动\lht.csv"
data = pd.read_csv(file_path)

# 参数设置
sampling_rate = 200  # 采样率：200 Hz
image_duration = 15  # 每张图像持续时间：15秒
points_per_image = sampling_rate * image_duration  # 每张图像的点数
num_images = 4  # 总图像数

# 验证数据是否足够
total_points = num_images * points_per_image
if len(data) < total_points:
    raise ValueError("数据长度不足，无法支持4张图像，每张图像15秒，采样率200Hz。")

# 计算全局的最大值和最小值，以确保色条一致
gaze_x = data['left_eye_gaze_position_x']
gaze_y = data['left_eye_gaze_position_y']
valid_mask = (gaze_x > 0) & (gaze_y > 0)  # 过滤无效凝视点
gaze_x = gaze_x[valid_mask]
gaze_y = gaze_y[valid_mask]
heatmap, xedges, yedges = np.histogram2d(
    gaze_x, gaze_y, bins=[50, 28], range=[[0, 1920], [0, 1080]]
)
global_vmax = np.max(heatmap)  # 全局最大值
global_vmin = 1  # 全局最小值

# 带背景图的热力图可视化
def plot_heatmap_with_background(data, title, background_img, save_path, vmin, vmax):
    plt.figure(figsize=(12, 6))  # 调整图形大小，匹配1920x1080的屏幕比例
    
    # 提取有效的X和Y凝视点
    gaze_x = data['bino_eye_gaze_position_x']
    gaze_y = data['bino_eye_gaze_position_y']
    valid_mask = (gaze_x > 0) & (gaze_y > 0)  # 过滤无效凝视点
    gaze_x = gaze_x[valid_mask]
    gaze_y = gaze_y[valid_mask]
    
    # 创建2D直方图作为热图
    heatmap, xedges, yedges = np.histogram2d(
        gaze_x, gaze_y, bins=[50, 28], range=[[0, 1920], [0, 1080]]
    )  # 调整箱子数目，保持矩形屏幕比例
    
    # 热图标准化，使用对数色条
    heatmap = heatmap.T  # 转置以匹配y轴方向
    norm = LogNorm(vmin=vmin, vmax=vmax)  # 使用对数色条，保持四张图色条一致
    
    # 绘制背景图
    plt.imshow(background_img, extent=[0, 1920, 0, 1080], aspect='auto', alpha=0.8)
    
    # 绘制热图，使用viridis配色
    plt.imshow(heatmap, cmap='coolwarm', alpha=0.6, extent=[0, 1920, 0, 1080], interpolation='bilinear', norm=norm)
    
    # 添加标题和标签
    plt.title(title, fontsize=20, fontweight='bold', color='black', pad=20)
    plt.xlabel("X", fontsize=16)
    plt.ylabel("Y", fontsize=16)
    
    # 设置坐标轴数字
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 优化色条
    cbar = plt.colorbar()  # 添加色条
    cbar.set_label('凝视密度', fontsize=14)  # 设置色条标签
    cbar.ax.tick_params(labelsize=12)  # 设置色条刻度字体大小
    cbar.ax.yaxis.set_ticks_position('right')  # 色条刻度位置调整为右侧
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图像到指定路径
    plt.savefig(save_path, dpi=300)  # 保存为300 DPI的高清图像
    plt.close()  # 关闭当前图像，避免显示

# 绘制每张图像段的热图并保存
for i in range(num_images):
    # 加载对应的背景图像
    background_image_path = f"C:\\Users\\Lhtooo\\Desktop\\生理信号分析\\images\\拼接\\{i + 1}.jpg"
    background_image = np.array(Image.open(background_image_path))
    
    # 选择当前图像段的数据
    start_idx = i * points_per_image
    end_idx = start_idx + points_per_image
    segment_data = data.iloc[start_idx:end_idx]
    
    # 设置保存路径
    save_path = f"C:\\Users\\Lhtooo\\Desktop\\生理信号分析\\output\\热图\\tgm_{i + 1}_heatmap.png"
    
    # 绘制带背景的热图并保存
    plot_heatmap_with_background(segment_data, f"图像{i + 1}的热图", background_image, save_path, vmin=global_vmin, vmax=global_vmax)

    print(f"保存图像: {save_path}")
