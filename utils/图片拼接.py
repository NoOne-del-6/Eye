import os
from PIL import Image

# 定义路径
root_folder = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\images\筛选"
output_folder = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\images\拼接4"
os.makedirs(output_folder, exist_ok=True)

# 获取每个情绪文件夹的图片列表，按文件名顺序排列
folders = sorted([os.path.join(root_folder, f) for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))])
images_by_folder = [sorted(os.listdir(folder), key=lambda x: int(os.path.splitext(x)[0])) for folder in folders]

# 持续拼接，直到所有文件夹的图片都拼接完成
index = 0
while any(images_by_folder):
    # 创建1920x1080的六宫格背景
    output_image = Image.new('RGB', (1920, 1080), (255, 255, 255))  # 白色背景
    
    # 每次从每个文件夹取一张图片
    for i, folder_images in enumerate(images_by_folder):
        if folder_images:  # 如果当前文件夹还有图片
            image_name = folder_images.pop(0)  # 取出第一个图片
            image_path = os.path.join(folders[i], image_name)
            img = Image.open(image_path)
            img = img.resize((640, 540))  # 调整为640x540
            
            # 计算位置：每行3张图，2行，共6张图
            row = i // 3
            col = i % 3
            position = (col * 640, row * 540)
            output_image.paste(img, position)

    # 保存拼接的六宫格
    output_path = os.path.join(output_folder, f"{index + 1}.jpg")
    output_image.save(output_path)
    print(f"第{index + 1}张六宫格图片已保存到: {output_path}")
    index += 1
