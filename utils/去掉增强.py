import json
import re

# JSON文件的路径
input_file_path = r"C:\Users\Lhtooo\Desktop\train+test_mod_1.json"
output_file_path = r"C:\Users\Lhtooo\Desktop\1\train+test_mod_2.json"

# 读取整个JSON文件
try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
except json.JSONDecodeError as e:
    print(f"文件解码失败: {e}")
    data = []  # 如果解码失败，设为空列表

# 准备一个新的列表来存储更新后的数据
filtered_data = []
deleted_images = []  # 存储被删除的图片文件名

# 遍历数据
for item in data:
    # 使用.get()安全获取images列表，如果没有则返回空列表
    images_list = item.get('images', [])
    # 检查每个图片文件名是否包含仅有数字的文件名
    keep_item = True  # 默认保留这个条目
    for image_path in images_list:
        # 提取文件名（去掉路径和扩展名）
        file_name = image_path.split('/')[-1].split('.')[0]
        # 检查文件名是否仅由数字组成
        if re.fullmatch(r'\d+', file_name):
            deleted_images.append(image_path)  # 添加到删除列表
            keep_item = False  # 发现纯数字的文件名，决定不保留这个条目

    # 如果没有找到仅数字的文件名，则保留这个条目
    if keep_item:
        filtered_data.append(item)

# 将处理后的数据写入新文件
with open(output_file_path, 'w', encoding='utf-8') as new_file:
    json.dump(filtered_data, new_file, ensure_ascii=False, indent=4)  # 美化输出


print("原来一共有{}个文件".format(len(data)))
print(f"共删除了{len(deleted_images)}个文件")
print(f"现在还剩下{len(filtered_data)}个文件")
print("处理后的文件已保存")
