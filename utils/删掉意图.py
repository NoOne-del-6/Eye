import json

# JSON文件的路径
input_file_path = r"D:\download\wechat\WeChat Files\wxid_3009970099711\FileStorage\File\2024-12\train+test_mod_1.jsonl"
output_file_path = r"C:\Users\Lhtooo\Desktop\train+test_mod_1.json"
# 读取并解析整个JSON文件
try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        # 假设整个文件是一个JSON数组
        data = json.load(file)
except json.JSONDecodeError as e:
    print(f"文件解码失败: {e}")
    data = []  # 如果解码失败，设为空列表

# 筛选含有"Picture"的query字段的数据
filtered_data = [item for item in data if "Picture" in item.get('query', '')]

# 如果找到包含"Picture"的条目，将它们写入到新的JSON文件中
if filtered_data:
    with open(output_file_path, 'w', encoding='utf-8') as new_file:
        json.dump(filtered_data, new_file, ensure_ascii=False, indent=4)  # 美化输出
    print("筛选后的文件已保存")
else:
    print("没有找到包含'Picture'关键字的条目或文件为空")
