import os
import sys
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QSizePolicy, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter


class Xieshi(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        # 用于存储 CSV 数据
        self.data = None
        self.blink_count = 0
        self.num_images = 0
        
        
        
    def initUI(self):
        self.setWindowIcon(QIcon(r'C:\Users\Lhtooo\Desktop\生理信号分析\code\images\bg.jpg'))  # 设置窗口图标
        self.setWindowTitle("眼动工具箱")
        self.setGeometry(300, 150, 1100, 800)

        # 设置窗口大小策略为可调整大小
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 设置 UI 布局
        layout = QVBoxLayout()

        # 添加文件加载按钮
        self.load_button = QPushButton('加载 CSV 文件')
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        # 添加数据处理按钮
        self.process_button = QPushButton('处理数据')
        self.process_button.clicked.connect(self.process_data)
        self.process_button.setEnabled(False)  # 初始状态不可用
        layout.addWidget(self.process_button)

        button_layout = QHBoxLayout()
        
        self.gaze1_button = QPushButton('注视稳定性')
        self.gaze1_button.clicked.connect(self.gaze_stability)
        self.gaze1_button.setEnabled(True)
        button_layout.addWidget(self.gaze1_button)
        
        self.gaze2_button = QPushButton('跳转反应时')
        self.gaze2_button.clicked.connect(self.tiaozhuan)
        self.gaze2_button.setEnabled(True)
        button_layout.addWidget(self.gaze2_button)
        
        self.gaze3_button = QPushButton('注视点可视化')
        self.gaze3_button.clicked.connect(self.gaze_vis)
        self.gaze3_button.setEnabled(True)
        button_layout.addWidget(self.gaze3_button)
        
        self.gaze4_button = QPushButton('热图分析')
        self.gaze4_button.clicked.connect(self.heatmap)
        self.gaze4_button.setEnabled(True)
        button_layout.addWidget(self.gaze4_button)
        
        self.gaze5_button = QPushButton('差异分析')
        self.gaze5_button.clicked.connect(self.chayi)
        self.gaze5_button.setEnabled(True)
        button_layout.addWidget(self.gaze5_button)
        
        layout.addLayout(button_layout)
        
      
        self.stats_button = QPushButton('统计量')
        self.stats_button.clicked.connect(self.show_statistics)
        self.stats_button.setVisible(False)
        self.stats_button.setEnabled(True)  # 初始状态不可用
        layout.addWidget(self.stats_button)
        
        # 添加显示消息的标签
        self.message_label = QLabel('请选择 眼动 文件进行加载')
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)
        
        # 添加返回主界面按钮
        self.back_button = QPushButton('返回主界面')
        self.back_button.clicked.connect(self.back_to_main)
        self.back_button.setVisible(False)
        layout.addWidget(self.back_button)
        
        
        # 添加 Matplotlib 图形画布
        self.canvas = FigureCanvas(Figure(figsize=(10, 8)))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
   
        
    def back_to_main(self):
        from main import MainApp
        self.close()  # 关闭当前窗口
        self.parent().setCentralWidget(MainApp())
        
        
    def load_csv(self):
        # 打开文件对话框，选择 CSV 文件
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 CSV 文件", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                # 读取 CSV 文件
                self.data = pd.read_csv(file_path)
                self.message_label.setText(f"文件加载成功：{file_path}")
                self.process_button.setEnabled(True)  # 文件加载成功后启用处理按钮
                self.num_images = len(self.data.index[self.data['trigger'] != 0].tolist())  
                

            except Exception as e:
                self.message_label.setText(f"加载失败：{str(e)}")
                self.data = None


    def process_data(self):
        if self.data is not None:
            try:
                original_length = len(self.data)
                # 删除无效数据
                self.data = self.data[(self.data['left_eye_valid'] != 0) &
                                      (self.data['right_eye_valid'] != 0) &
                                      (self.data['bino_eye_valid'] != 0)]
                removed_length = original_length - len(self.data)
                self.message_label.setText(f"数据预处理完成, 去除无效数据记录数: {removed_length}")
                # self.pupil_button.setEnabled(True)  # 处理完成后启用绘图按钮
                # self.heatmap_button.setEnabled(True)
                # self.emotion_button.setEnabled(True)
            except Exception as e:
                self.message_label.setText(f"数据处理失败：{str(e)}")
                self.data = None

    def gaze_stability(self):
        if self.data is not None:
            try:
                trigger = self.data['trigger']
                # 设定距离阈值为150像素
                distance_threshold = 150

                # 计算双眼注视点一致性（欧几里得距离）
                def euclidean_distance(x1, y1, x2, y2):
                    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # 获取所有触发器为 202 的索引值，标记图像分割位置
                trigger_indices = np.where(trigger == 202)[0]

                # 用于存储每张图像的分析结果
                image_data = []

                # 用于计算异常帧数
                consecutive_abnormal_frames = 0
                max_consecutive_abnormal_frames = 0
                total_abnormal_frames = 0  # 用于计算总的异常帧数

                # 为每张图像进行分析
                start_idx = 0
                for idx in trigger_indices:
                    image_segment = self.data[start_idx:idx]  # 当前图像的注视数据
                    start_idx = idx + 1  # 下一张图像的开始位置
                    
                    # 提取当前图像的注视点
                    left_eye_x_img = image_segment['left_eye_gaze_position_x']
                    left_eye_y_img = image_segment['left_eye_gaze_position_y']
                    right_eye_x_img = image_segment['right_eye_gaze_position_x']
                    right_eye_y_img = image_segment['right_eye_gaze_position_y']
                    # 计算双眼注视点一致性（欧几里得距离）
                    distances_img = euclidean_distance(left_eye_x_img, left_eye_y_img, right_eye_x_img, right_eye_y_img)
                    abnormal_gaze_points_img = distances_img > distance_threshold

                    # 计算左右眼注视点稳定性（标准差）
                    std_left_x = np.std(left_eye_x_img)
                    std_left_y = np.std(left_eye_y_img)
                    std_right_x = np.std(right_eye_x_img)
                    std_right_y = np.std(right_eye_y_img)

                    # 生成每张图像的注视点分析结果
                    image_data.append({
                        'Image Index': len(image_data) + 1,
                        'Left Eye Std X': std_left_x,
                        'Left Eye Std Y': std_left_y,
                        'Right Eye Std X': std_right_x,
                        'Right Eye Std Y': std_right_y,
                    })
                    
                    # 统计连续异常帧数
                    abnormal_gaze_points_img = abnormal_gaze_points_img.values if isinstance(abnormal_gaze_points_img, pd.Series) else abnormal_gaze_points_img  # 确保为NumPy数组
                    consecutive_abnormal_frames = 0  # 重置连续异常帧计数器
                    for i in range(len(abnormal_gaze_points_img)):  # 使用len()保证索引不超出范围
                        if abnormal_gaze_points_img[i]:
                            consecutive_abnormal_frames += 1
                        else:
                            max_consecutive_abnormal_frames = max(max_consecutive_abnormal_frames, consecutive_abnormal_frames)
                            consecutive_abnormal_frames = 0

                    # 累计异常帧数
                    total_abnormal_frames += np.sum(abnormal_gaze_points_img)

                max_consecutive_abnormal_frames = max(max_consecutive_abnormal_frames, consecutive_abnormal_frames)


                # 将每张图像的分析结果转换为 DataFrame
                image_df = pd.DataFrame(image_data)


                # 可视化图像的注视点稳定性柱状图（左眼与右眼）
                self.canvas.figure.clear()
                ax = self.canvas.figure.add_subplot(111)

                # 设定条形图宽度和间隔
                bar_width = 0.35
                index = np.arange(len(image_data))

                # 绘制左眼和右眼的标准差柱状图
                ax.bar(index, image_df['Left Eye Std X'], bar_width, color='pink', label='Left Eye X Std')
                ax.bar(index + bar_width, image_df['Right Eye Std X'], bar_width, color='green', label='Right Eye X Std')

                # 添加标题和标签
                ax.set_title("Stability of Eye Gaze Points per Image (Left vs Right Eye)", fontsize=16)
                ax.set_xlabel("Image Index", fontsize=12)
                ax.set_ylabel("Standard Deviation (X)", fontsize=12)
                ax.set_xticks(index + bar_width / 2)
                ax.set_xticklabels([f"Image {i+1}" for i in range(len(image_data))])  # 设置图像标签
                ax.legend()

                # 美化图表：设置网格、背景和文本大小
                ax.grid(True, linestyle='--', alpha=0.7)
                self.canvas.draw_idle()
                self.message_label.setText(f"注视稳定性分析成功")
            except Exception as e:
                self.message_label.setText(f"注视稳定性分析失败：{str(e)}")

    def tiaozhuan(self):
        if self.data is not None:
            try:
                # 假设目标变化时的触发器（trigger == 202）标记了视觉任务的目标变化
                target_change_indices = np.where(self.data['trigger'] == 202)[0]

                # 采样频率 200Hz，即每帧的时间间隔为 0.005 秒（5 毫秒）
                sampling_interval = 0.005  # 每帧的时间间隔为 5 毫秒

                # 存储每张图片跳转的反应时间
                reaction_times_per_image = []

                # 记录每张图片的反应时间
                for i in range(1, len(target_change_indices)):
                    # 获取当前目标变化的时间戳（目标变化时间）
                    target_change_time = self.data['timestamp'].iloc[target_change_indices[i]]
                    
                    # 假设时间单位为毫微秒（nanoseconds），转换为秒（除以 1e9）
                    target_change_time = target_change_time / 1e9  # 转换为秒（如果单位是毫微秒）
                    
                    # 获取目标变化之前的注视点（注视点位置）
                    previous_x = self.data['bino_eye_gaze_position_x'].iloc[target_change_indices[i]]
                    previous_y = self.data['bino_eye_gaze_position_y'].iloc[target_change_indices[i]]
                    
                    # 遍历数据，寻找眼动开始的时刻
                    start_reaction_time = None
                    for j in range(target_change_indices[i], len(self.data) - 1):
                        current_x = self.data['bino_eye_gaze_position_x'].iloc[j]
                        current_y = self.data['bino_eye_gaze_position_y'].iloc[j]
                        
                        # 计算位移（欧几里得距离）
                        displacement = np.sqrt((current_x - previous_x)**2 + (current_y - previous_y)**2)
                        
                        # 设置一个位移阈值，假设100像素为合适的值
                        displacement_threshold = 100
                        
                        # 如果位移超过阈值，认为眼动开始
                        if displacement > displacement_threshold:
                            start_reaction_time = self.data['timestamp'].iloc[j]
                            start_reaction_time = start_reaction_time / 1e9  # 转换为秒
                            break
                        
                        # 更新前一个位置
                        previous_x, previous_y = current_x, current_y
                    
                    # 如果找到了眼动开始的时间，则计算反应时间
                    if start_reaction_time:
                        reaction_time = start_reaction_time - target_change_time
                        # 增加最小反应时间阈值，确保反应时间为合理的正数（例如5毫秒，即0.005秒）
                        if reaction_time > sampling_interval:  # 设置最小反应时间阈值为5毫秒
                            reaction_times_per_image.append(reaction_time)

                # 如果需要计算平均反应时间
                if reaction_times_per_image:
                    average_reaction_time = np.mean(reaction_times_per_image)
                    print(f"所有图片的平均反应时间：{average_reaction_time:.4f} 秒")
                else:
                    print("没有有效的反应时间数据")

                # 可视化每张图片的反应时间
                self.canvas.figure.clear()
                ax = self.canvas.figure.add_subplot(111)
                ax.bar(range(1, len(reaction_times_per_image) + 1), reaction_times_per_image, color='skyblue')

                # 设置图表标题和标签
                ax.set_title('Reaction Time per Image', fontsize=16)
                ax.set_xlabel('Image Index', fontsize=12)
                ax.set_ylabel('Reaction Time (seconds)', fontsize=12)
                ax.set_xticks(range(1, len(reaction_times_per_image) + 1))  # 设置X轴标签为图片编号
                ax.grid(True, linestyle='--', alpha=0.7)

                # 设置y轴范围，保证文本不超出图形区域
                ax.set_ylim(0, max(reaction_times_per_image) + 0.2)  # 给反应时间加一点空间，防止文本超出

                # 在每个柱状图上方显示反应时间
                for idx, reaction_time in enumerate(reaction_times_per_image, 1):
                    ax.text(idx, reaction_time + 0.05, f'{reaction_time:.4f}', ha='center', fontsize=10)

                # 显示柱状图
                self.canvas.draw_idle()
                self.message_label.setText(f"跳转反应时计算成功")

            except Exception as e:
                self.message_label.setText(f"跳转反应时计算失败：{str(e)}")

    def heatmap(self):
        if self.data is not None:
            try:
                # 加载图像文件
                image_path = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\images\xieshi.png"
                image = mpimg.imread(image_path)

                # 获取图像尺寸
                img_height, img_width, _ = image.shape

                # 获取左右眼的注视点数据
                left_eye_x = self.data['left_eye_gaze_position_x']
                left_eye_y = self.data['left_eye_gaze_position_y']
                right_eye_x = self.data['right_eye_gaze_position_x']
                right_eye_y = self.data['right_eye_gaze_position_y']

                # 过滤掉超出图像范围的注视点
                left_eye_x = left_eye_x[(left_eye_x > 0) & (left_eye_x < img_width) & (left_eye_y > 0) & (left_eye_y < img_height)]
                left_eye_y = left_eye_y[(left_eye_x > 0) & (left_eye_x < img_width) & (left_eye_y > 0) & (left_eye_y < img_height)]
                right_eye_x = right_eye_x[(right_eye_x > 0) & (right_eye_x < img_width) & (right_eye_y > 0) & (right_eye_y < img_height)]
                right_eye_y = right_eye_y[(right_eye_x > 0) & (right_eye_x < img_width) & (right_eye_y > 0) & (right_eye_y < img_height)]

                # 创建热图数据
                heatmap_left, xedges_left, yedges_left = np.histogram2d(left_eye_x, left_eye_y, bins=100, range=[[0, img_width], [0, img_height]])
                heatmap_right, xedges_right, yedges_right = np.histogram2d(right_eye_x, right_eye_y, bins=100, range=[[0, img_width], [0, img_height]])

                # 高斯模糊处理
                heatmap_left = gaussian_filter(heatmap_left, sigma=3)
                heatmap_right = gaussian_filter(heatmap_right, sigma=3)

                # 可视化热图
                self.canvas.figure.clear()
                ax = self.canvas.figure.add_subplot(111)
                ax.imshow(image, extent=[0, img_width, 0, img_height], origin='upper')  # 显示背景图像
                ax.imshow(heatmap_left.T, extent=[0, img_width, 0, img_height], origin='upper', cmap='Blues', alpha=0.6)  # 左眼热图
                ax.imshow(heatmap_right.T, extent=[0, img_width, 0, img_height], origin='upper', cmap='Reds', alpha=0.6)  # 右眼热图

                # 设置标题和标签
                ax.set_title("Heatmap of Eye Gaze Points", fontsize=16)
                ax.set_xlabel("X Position (pixels)", fontsize=12)
                ax.set_ylabel("Y Position (pixels)", fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)

                # 更新画布
                self.canvas.draw_idle()
                self.message_label.setText("热图分析成功")
            except Exception as e:
                self.message_label.setText(f"热图分析失败：{str(e)}")
        
    def gaze_vis(self):
        if self.data is not None:
            try:
                # 获取左右眼的注视点数据
                left_eye_x = self.data['left_eye_gaze_position_x']
                left_eye_y = self.data['left_eye_gaze_position_y']
                right_eye_x = self.data['right_eye_gaze_position_x']
                right_eye_y = self.data['right_eye_gaze_position_y']

                # 创建一个新的图形
                self.canvas.figure.clear()
                ax = self.canvas.figure.add_subplot(111)

                # 绘制左眼和右眼的注视点
                ax.scatter(left_eye_x, -left_eye_y, c='blue', label='Left Eye', alpha=0.6, s=10)
                ax.scatter(right_eye_x, -right_eye_y, c='red', label='Right Eye', alpha=0.6, s=10)

                # 设置标题和标签
                ax.set_title("Visualization of Eye Gaze Points", fontsize=16)
                ax.set_xlabel("X Position (pixels)", fontsize=12)
                ax.set_ylabel("Y Position (pixels)", fontsize=12)
                ax.legend()

                # 显示网格
                ax.grid(True, linestyle='--', alpha=0.7)

                # 更新画布
                self.canvas.draw_idle()
                self.message_label.setText("注视点可视化成功")
            except Exception as e:
                self.message_label.setText(f"注视点可视化失败：{str(e)}")

    def chayi(self):
        if self.data is not None:
            try:
                # 计算左右眼视线位置的欧几里得距离
                x_diff = self.data['left_eye_gaze_position_x'] - self.data['bino_eye_gaze_position_x']
                y_diff = self.data['left_eye_gaze_position_y'] - self.data['bino_eye_gaze_position_y']
                euclidean_distance = np.sqrt(x_diff**2 + y_diff**2)

                # 定义异常视线点的阈值（150像素）
                threshold = 150

                # 创建图表
                self.canvas.figure.clear()
                ax = self.canvas.figure.add_subplot(111)
                ax.plot(euclidean_distance, color='lightblue', label="Euclidean Distance")
                ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold} pixels)')

                # 给图表添加标题和标签
                abnormal_ratio = np.sum(euclidean_distance > threshold) / len(euclidean_distance)
                ax.set_title(f'Euclidean Distance Between Eye Gaze Points\nAbnormal Gaze Points Ratio: {abnormal_ratio:.4f} ({np.sum(euclidean_distance > threshold)}/{len(euclidean_distance)})')
                ax.set_xlabel('Frame Index')
                ax.set_ylabel('Distance (pixels)')
                ax.legend()

                # 显示图表
                ax.grid(True, linestyle='--', alpha=0.7)
              
                self.canvas.draw_idle()
                self.message_label.setText("差异分析成功")
            except Exception as e:
                self.message_label.setText(f"差异分析失败：{str(e)}")
 
    def show_statistics(self):
        if self.data is not None:
            try:
                # 提取左右眼和双眼的注视点
                gaze_x = self.data['bino_eye_gaze_position_x']  # 双眼注视点X
                gaze_y = self.data['bino_eye_gaze_position_y']  # 双眼注视点Y

                # 设定距离阈值为150像素
                distance_threshold = 150

                # 在计算距离之前，确保数据没有缺失（NaN）
                self.data['distance'] = np.sqrt(
                                        (gaze_x.diff())**2 +
                                        (gaze_y.diff())**2
                                    ).fillna(0)  # 填充NaN为0，防止计算错误
                
                # 标记双眼注视点异常的帧
                abnormal_gaze_points = self.data['distance'] > distance_threshold

                # 2. 计算注视点稳定性（标准差）
                # 计算双眼注视点的标准差，确保去掉NaN值
                std_bino_x = np.std(gaze_x.dropna())  # 删除缺失数据
                std_bino_y = np.std(gaze_y.dropna())  # 删除缺失数据

                # 3. 时间维度的注视行为
                # 设定连续异常帧的阈值（例如超过100毫秒，即10帧，假设帧率为100Hz）
                frame_rate = 100  # 假设的帧率，100帧/秒

                # 统计连续异常帧数
                consecutive_abnormal_frames = 0
                max_consecutive_abnormal_frames = 0
                for i in range(1, len(abnormal_gaze_points)):
                    if abnormal_gaze_points[i]:
                        consecutive_abnormal_frames += 1
                    else:
                        max_consecutive_abnormal_frames = max(max_consecutive_abnormal_frames, consecutive_abnormal_frames)
                        consecutive_abnormal_frames = 0

                max_consecutive_abnormal_frames = max(max_consecutive_abnormal_frames, consecutive_abnormal_frames)
                
                # 生成优化后的统计字符串
                statistics_str = (
                    f"<div style='font-family: Arial, sans-serif; color: #333; font-size: 16px; line-height: 1.4;'>"
                    f"<b><font size='5' color='#4A90E2'>总体注视点稳定性：标准差 (X)  : </font></b> <font size='5' color='#333333'>{std_bino_x:.2f}</font><br>"
                    f"<b><font size='5' color='#4A90E2'>总体注视点稳定性：标准差 (Y)  : </font></b> <font size='5' color='#333333'>{std_bino_y:.2f}</font><br>"
                    f"<b><font size='5' color='#4A90E2'>总体双眼注视点一致性：异常注视点数量（大于150像素距离）:</font></b> <font size='5' color='#333333'>{np.sum(abnormal_gaze_points)}</font><br>"
                    f"<b><font size='5' color='#4A90E2'>总体时间维度的注视行为：最大连续异常帧数：</font></b> <font size='5' color='#333333'>{max_consecutive_abnormal_frames}</font><br>"
                    f"<font size='4' color='gray' style='font-size: 14px;'>统计完毕，数据已更新。</font>"
                    f"</div>"
                )

                self.canvas.figure.clf()
                self.canvas.draw_idle()
                # 设置更新消息
                self.message_label.setText(statistics_str)

            except Exception as e:
                self.message_label.setText(f"<font color='red' style='font-size: 14px;'><b>统计量计算失败：</b>{str(e)}</font>")






if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = Xieshi()
    main_window.show()
    sys.exit(app.exec_())
