# 将原始文件的时间戳转换为系统时间
import datetime
import os

import pandas as pd
import psutil

class CSVConvert:
    def __init__(self):
        # 获取电脑启动时间（时间戳，秒）
        self.boot_time_timestamp = int(psutil.boot_time())

    # 假设时间戳列名为 "timestamp_ns"，你可以根据实际的列名修改
    def convert_timestamp_to_system_time(self, timestamp_ns):
        # 将纳秒时间戳转换为秒
        timestamp_seconds = int(timestamp_ns / 1_000_000_000)

        # # 计算运行时间（即时间戳对应的秒数）
        # uptime = datetime.timedelta(seconds=timestamp_seconds)

        # 计算系统时间（启动时间 + 运行时间）
        system_time = self.boot_time_timestamp + timestamp_seconds

        # # 获取纳秒部分
        # nanoseconds = timestamp_ns % 1_000_000_000
        #
        # formatted_time_with_ns = f"{system_time}.{nanoseconds:09d}"  # 添加纳秒部分

        return system_time

    def convert(self, csv_filename):
    # 读取CSV文件
    #     file_name = 'a.csv'
    #     csv_filename = os.path.join(data_dir, file_name)  # 替换为你的文件名
        df = pd.read_csv(csv_filename)

        # 使用apply方法将新列添加到DataFrame中
        df['real_system_time'] = df['timestamp'].apply(self.convert_timestamp_to_system_time)

        _dir, _filename = os.path.split(csv_filename)

        basename = os.path.splitext(_filename)[0]
        extension = os.path.splitext(_filename)[1]


        _csv_name = f"{basename}_new{extension}"
        _csv_path = os.path.join(_dir, _csv_name)
        # 将新的数据保存到原CSV文件中
        df.to_csv(_csv_path, index=False)


if __name__ == "__main__":
    con = CSVConvert()
    con.convert(r'ceshi32male_20250325_GAD-7广泛性焦虑障碍量表.csv')