import pandas as pd

input_file = r"C:\Users\Lhtooo\Desktop\same\835.csv"
output_file = r"C:\Users\Lhtooo\Desktop\same\835_utf8.csv"

try:
    # 尝试使用 GBK 读取文件
    df = pd.read_csv(input_file, encoding='gbk')
    df.to_csv(output_file, encoding='utf-8', index=False)
    print(f"文件 {input_file} 转换成功！保存为 {output_file}")
except Exception as e:
    print(f"文件转换失败：{e}")
