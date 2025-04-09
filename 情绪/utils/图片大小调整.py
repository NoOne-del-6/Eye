
from PIL import Image
import os

# 文件夹路径
folder_path = r'D:\working\生理信号分析\Emotion6\images\disgust'  # 替换为你的文件夹路径
output_folder = r'D:\working\生理信号分析\Emotion6\images\disgust_1'  # 输出文件夹

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 目标分辨率
target_size = (1920, 1080)

# 批量调整图片大小
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # 支持的图片格式
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        
        # 调整大小并保持比例
        img = img.resize(target_size, Image.LANCZOS)
        
        # 保存调整后的图片
        img.save(os.path.join(output_folder, filename))

print("图片大小已全部调整为 1920x1080！")
