import pandas as pd


def compare_csv_files(csv_path1, csv_path2, csv_path3):
    # 读取三个 CSV 文件
    df1 = pd.read_csv(csv_path1,encoding='utf-8')
    df2 = pd.read_csv(csv_path2,encoding='utf-8')
    df3 = pd.read_csv(csv_path3,encoding='utf-8')

    # 确保三个文件都包含 'id' 列
    if 'id' not in df1.columns or 'id' not in df2.columns or 'id' not in df3.columns:
        print("三个文件必须包含 'id' 列")
        return

    # 重命名列以区分
    df1 = df1.rename(columns={'predict': 'predict_1'})
    df2 = df2.rename(columns={'predict': 'predict_2'})
    df3 = df3.rename(columns={'predict': 'predict_3'})

    # 合并数据集
    merged_df = pd.merge(df1, df2, on='id', how='inner')
    merged_df = pd.merge(merged_df, df3, on='id', how='inner')

    # 比较 predict 字段
    mismatched = merged_df[(merged_df['predict_1'] != merged_df['predict_2']) |
                           (merged_df['predict_1'] != merged_df['predict_3']) |
                           (merged_df['predict_2'] != merged_df['predict_3'])]

    matched = merged_df[(merged_df['predict_1'] == merged_df['predict_2']) &
                        (merged_df['predict_1'] == merged_df['predict_3'])]

    print(f"Predict 字段不一样的数量: {len(mismatched)}")
    print(f"Predict 字段一样的数量: {len(matched)}")
    print("\nPredict 不一致的详细结果:")
    print(mismatched[['id', 'predict_1', 'predict_2', 'predict_3']])

    # 将结果保存到文件
    mismatched.to_csv(r"C:\Users\Lhtooo\Desktop\1\no_same.csv", index=False,
                      encoding='utf-8-sig')
    print("不一致的结果已保存到 mismatched_results.csv")


if __name__ == "__main__":
    # CSV 文件路径
    csv_file_1 = r"C:\Users\Lhtooo\Desktop\1\output_gbk_1.csv"
    csv_file_2 = r"C:\Users\Lhtooo\Desktop\1\output_gbk_2.csv"
    csv_file_3 = r"C:\Users\Lhtooo\Desktop\1\output_gbk_3.csv"

    # 执行比较
    compare_csv_files(csv_file_1, csv_file_2, csv_file_3)
