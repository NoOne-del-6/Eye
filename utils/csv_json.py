# import pandas as pd
# import json

# def csv_to_json(input_csv, output_json):
#     """
#     将整个CSV文件转换为JSON文件
#     """
#     try:
#         # 读取CSV文件
#         df = pd.read_csv(input_csv)

#         # 将DataFrame转换为字典格式（每行一个JSON对象）
#         data = df.to_dict(orient='records')

#         # 将字典格式的数据写入JSON文件
#         with open(output_json, 'w', encoding='utf-8') as f:
#             json.dump(data, f, ensure_ascii=False, indent=4)

#         print(f"CSV文件已成功转换为JSON文件，输出保存为: {output_json}")
#     except Exception as e:
#         print(f"转换失败: {e}")

# # 输入和输出文件路径
# input_csv = r"C:\Users\Lhtooo\Desktop\same\same.csv"     # 替换为您的CSV文件路径
# output_json = r"C:\Users\Lhtooo\Desktop\same\same.json"  # 替换为您希望保存的JSON文件路径

# # 执行转换
# csv_to_json(input_csv, output_json)

# import json

# def process_json_image_field(input_json, output_json):
#     """
#     处理 JSON 文件中的 `image` 字段，修改为新的格式，并保存到新文件。
#     """
#     try:
#         # 读取 JSON 文件
#         with open(input_json, 'r', encoding='utf-8') as f:
#             data = json.load(f)

#         # 遍历每个 JSON 对象，修改 `image` 字段
#         for item in data:
#             if 'image' in item and isinstance(item['image'], str):
#                 # 提取 image 名称并修改格式
#                 raw_images = eval(item['image'])  # 将字符串解析为列表
#                 new_images = [f"images/{image}" for image in raw_images]
#                 item['image'] = new_images

#         # 保存修改后的 JSON 文件
#         with open(output_json, 'w', encoding='utf-8') as f:
#             json.dump(data, f, ensure_ascii=False, indent=4)

#         print(f"处理完成，修改后的 JSON 文件已保存为：{output_json}")

#     except Exception as e:
#         print(f"处理失败：{e}")


# # 输入和输出文件路径
# input_json = r"C:\Users\Lhtooo\Desktop\same\same.json"      # 替换为您的输入 JSON 文件路径
# output_json = r"C:\Users\Lhtooo\Desktop\same\same_processed.json"  # 替换为处理后的输出 JSON 文件路径

# # 执行处理
# process_json_image_field(input_json, output_json)
# import json

# def replace_predict_with_output(input_json, output_json):
#     """
#     将JSON文件中的`predict`字段全部替换为`output`字段
#     """
#     try:
#         # 读取 JSON 文件
#         with open(input_json, 'r', encoding='utf-8') as f:
#             data = json.load(f)

#         # 遍历每个 JSON 对象，修改 `predict` 为 `output`
#         for item in data:
#             if 'predict' in item:
#                 item['output'] = item.pop('predict')  # 替换键名

#         # 保存修改后的 JSON 文件
#         with open(output_json, 'w', encoding='utf-8') as f:
#             json.dump(data, f, ensure_ascii=False, indent=4)

#         print(f"字段替换完成，修改后的 JSON 文件已保存为：{output_json}")

#     except Exception as e:
#         print(f"处理失败：{e}")

# # 输入和输出文件路径
# input_json = r"C:\Users\Lhtooo\Desktop\same\same.json"      # 替换为您的输入 JSON 文件路径
# output_json = r"C:\Users\Lhtooo\Desktop\same\same_modified.json"  # 替换为处理后的输出 JSON 文件路径

# # 执行处理
# replace_predict_with_output(input_json, output_json)



import os
import pandas as pd

# 设置文件夹路径
folder_path = r'C:\Users\Lhtooo\Desktop\same'

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # 尝试不同的编码格式读取文件
        encodings = ['utf-8', 'ISO-8859-1', 'gbk', 'windows-1252']
        for encoding in encodings:
            try:
                print(f"正在尝试使用 {encoding} 编码读取文件: {filename}")
                df = pd.read_csv(file_path, encoding=encoding)
                
                # 转换为 UTF-8-SIG 并覆盖原文件
                df.to_csv(file_path, encoding='utf-8-sig', index=False)
                print(f"{filename} 已成功转码为 UTF-8-SIG")
                break  # 成功读取后退出编码尝试
            except Exception as e:
                print(f"使用 {encoding} 编码读取时出错: {e}")

