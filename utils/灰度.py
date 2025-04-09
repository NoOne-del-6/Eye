from PIL import Image
import numpy as np

# 路径到图片文件
image_path = r'C:\Users\Lhtooo\Desktop\生理信号分析\code\data\6.jpg'

# 定义感兴趣区域 (AOI) 的边界
aoi_bounds = {
    "anger": [0, 640, 0, 540],
    "disgust": [640, 1280, 0, 540],
    "fear": [1280, 1920, 0, 540],
    "joy": [0, 640, 540, 1080],
    "sadness": [640, 1280, 540, 1080],
    "surprise": [1280, 1920, 540, 1080]
}

# 打开图片并转换为灰度
img = Image.open(image_path).convert('L')

# 初始化灰度矩阵
gray_matrix = np.zeros(len(aoi_bounds))

# 计算每个区域的平均灰度并存储到矩阵
for idx, (name, bounds) in enumerate(aoi_bounds.items()):
    # 提取每个区域的像素
    box = (bounds[0], bounds[2], bounds[1], bounds[3])  # 左，上，右，下
    region = img.crop(box)
    
    # 计算平均灰度
    mean_gray = sum(region.getdata()) / (region.size[0] * region.size[1])
    
    # 保存到矩阵
    gray_matrix[idx] = mean_gray

    # 打印结果
    print(f"{name} area average gray level: {mean_gray:.2f}")

# 打印完整的灰度矩阵
print("Gray level matrix:")
print(gray_matrix)
