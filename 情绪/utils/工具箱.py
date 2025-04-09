import os
import sys
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QHBoxLayout, QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter


class CsvVisualizerApp(QWidget):
    
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(r'C:\Users\Lhtooo\Desktop\生理信号分析\code\images\bg.jpg'))  # 设置窗口图标
        self.setWindowTitle("眼动工具箱")
        self.setGeometry(300, 150, 1100, 800)

        # 设置窗口大小策略为可调整大小
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 用于存储 CSV 数据
        self.data = None

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
        
        self.pupil_button = QPushButton('瞳孔可视化')
        self.pupil_button.clicked.connect(self.pupil_plot)
        self.pupil_button.setEnabled(False)  # 初始状态不可用
        button_layout.addWidget(self.pupil_button)
        self.heatmap_button = QPushButton('热图分析')
        self.heatmap_button.clicked.connect(self.heatmap)
        self.heatmap_button.setEnabled(False)  # 初始状态不可用
        button_layout.addWidget(self.heatmap_button)
        layout.addLayout(button_layout)
        
        # 添加情绪占比按钮和统计量按钮
        self.emotion_button = QPushButton('情绪占比')
        self.emotion_button.clicked.connect(self.emotion)
        self.emotion_button.setEnabled(False)  # 初始状态不可用
        button_layout.addWidget(self.emotion_button)
        
        self.toggle_plot_button = QPushButton("切换到雷达图")
        self.toggle_plot_button.setVisible(False)
        self.toggle_plot_button.clicked.connect(self.toggle_plot)
        self.toggle_plot_button.setEnabled(False)  # 初始状态不可用
        layout.addWidget(self.toggle_plot_button)
        self.plot_type='bar'
        
        self.stats_button = QPushButton('统计量')
        self.stats_button.clicked.connect(self.show_statistics)
        self.stats_button.setEnabled(True)  # 初始状态不可用
        layout.addWidget(self.stats_button)
        
        # 添加显示消息的标签
        self.message_label = QLabel('请选择 眼动 文件进行加载')
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)

        # 添加 Matplotlib 图形画布
        self.canvas = FigureCanvas(Figure(figsize=(10, 8)))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        # 定义感兴趣区域 (AOI) 的边界
        self.aoi_bounds = {
            "anger": [0, 640, 0, 540],        # Top-left
            "disgust": [640, 1280, 0, 540],   # Top-center
            "fear": [1280, 1920, 0, 540],     # Top-right
            "joy": [0, 640, 540, 1080],       # Bottom-left
            "sadness": [640, 1280, 540, 1080], # Bottom-center
            "surprise": [1280, 1920, 540, 1080] # Bottom-right
        }
        self.blink_count = 0
        self.num_images = 0
        
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
                 # 统计眨眼次数
                blink_count = 0
                in_blink = False  # 记录是否处于眨眼状态

                for i in range(1, len(self.data)):
                    # 判断是否为连续无效数据（valid = 0）
                    if self.data['left_eye_valid'][i] == 0 and self.data['left_eye_valid'][i - 1] != 0:
                        in_blink = True  # 进入眨眼状态
                    elif self.data['left_eye_valid'][i] != 0 and in_blink:
                        blink_count += 1  # 如果从眨眼状态变为有效数据，统计一次眨眼
                        in_blink = False  # 结束眨眼状态

                # 保存眨眼次数到全局变量
                self.blink_count = blink_count  
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
                self.pupil_button.setEnabled(True)  # 处理完成后启用绘图按钮
                self.heatmap_button.setEnabled(True)
                self.emotion_button.setEnabled(True)
            except Exception as e:
                self.message_label.setText(f"数据处理失败：{str(e)}")
                self.data = None

    def plot_combined_features(self, data, features, title, ylabel, feature_labels):
            sampling_rate = 200  # Hz
            image_duration = 10  # seconds
            points_per_image = sampling_rate * image_duration  # Data points per image
            num_images = self.num_images # Total number of images

            # 获取当前 figure 和 ax 对象
            fig = self.canvas.figure  # 使用当前的 figure
            ax = fig.add_subplot(111)  # 创建一个新的子图，并加入到当前 figure 中

            # 清除旧的内容
            ax.clear()

    
            trigger_indices = self.data.index[self.data['trigger'] != 0].tolist()
            
            if not trigger_indices:
                self.message_label.setText("未找到有效的触发器数据")
                return

            # Start plotting from the first trigger
            start_idx = trigger_indices[0]
            
            for i in range(num_images):
                if i < len(trigger_indices) - 1:
                    end_idx = trigger_indices[i + 1]
                else:
                    end_idx = start_idx + points_per_image

                segment_data = data.iloc[start_idx:end_idx]
                
                # Create a relative time axis
                timestamps_relative = (segment_data.index - start_idx) / sampling_rate  # Seconds
                
                # Plot the combined features for this image
                for feature, label in zip(features, feature_labels):
                    ax.plot(
                        timestamps_relative + i * image_duration,  # Offset each image by its starting time
                        segment_data[feature], 
                        label=f'Image {i + 1} - {label}', 
                        alpha=0.7
                    )
                
              
                if i > 0:
                    ax.axvline(x=i * image_duration, color='red', linestyle='--')
                
               
                start_idx = end_idx

            # Add labels, title, and legend
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            # 强制更新图形
            self.canvas.draw_idle()  # 强制更新图形

    def pupil_plot(self):
        if self.data is not None:
            try:
                self.toggle_plot_button.setVisible(False)
                # 清除现有图形
                self.canvas.figure.clear()
                self.plot_combined_features(
                    self.data,
                    ['left_eye_pupil_diameter_mm', 'right_eye_pupil_diameter_mm'],
                    'Pupil Diameter (Left and Right) Over Time',
                    'Pupil Diameter (mm)',
                    ['Left Pupil', 'Right Pupil']
                )

                # 强制更新图形
                self.canvas.draw_idle()  # 这个方法相对于 `draw()` 更加高效且适用于在事件处理期间刷新画布
                self.message_label.setText("瞳孔可视化成功")
            except Exception as e:
                self.message_label.setText(f"瞳孔可视化失败：{str(e)}")
    

    def heatmap(self):
        if self.data is not None:
            try:
                self.toggle_plot_button.setVisible(False)
                # 清除现有图形
                self.canvas.figure.clear()
                # 获取canvas的宽度和高度
                canvas_width = self.canvas.size().width()
                canvas_height = self.canvas.size().height()

                # 使用canvas的尺寸来设置figsize
                dpi = self.canvas.figure.dpi  # 获取当前figure的dpi
                figsize = (canvas_width / dpi, canvas_height / dpi)  # 计算figsize

                # 创建2x2的子图，使用适配画布的尺寸
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, constrained_layout=True)
                axes = axes.flatten()

                # 获取程序运行时的路径，处理打包后的路径
                base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))

                # 构建图片的路径
                background_image_paths = [
                    os.path.join(base_path, "拼接", "1.jpg"),  # 第一个背景图
                    os.path.join(base_path, "拼接", "2.jpg"),  # 第二个背景图
                    os.path.join(base_path, "拼接", "3.jpg"),  # 第三个背景图
                    os.path.join(base_path, "拼接", "4.jpg")   # 第四个背景图
                ]
                
                trigger_indices = self.data.index[self.data['trigger'] != 0].tolist()

                if not trigger_indices:
                    self.message_label.setText("未找到有效的触发器数据")
                    return

                sampling_rate = 200  # Hz
                image_duration = 15  # seconds
                points_per_image = sampling_rate * image_duration  # Data points per image

                start_idx = trigger_indices[0]

                for i, ax in enumerate(axes):
                    if i < len(trigger_indices) - 1:
                        end_idx = trigger_indices[i + 1]
                    else:
                        end_idx = start_idx + points_per_image

                    segment_data = self.data.iloc[start_idx:end_idx]
                    # Create heatmap data for gaze positions
                    heatmap_data, xedges, yedges = np.histogram2d(
                        segment_data['bino_eye_gaze_position_x'],
                        segment_data['bino_eye_gaze_position_y'],
                        bins=[50, 28],
                        range=[[0, 1920], [0, 1080]]
                    )

                    # 使用高斯滤波进行平滑处理
                    smoothed_heatmap = gaussian_filter(heatmap_data, sigma=0.8)  # 增加sigma值增加平滑效果

                    # 根据子图索引选择背景图
                    background_img_path = background_image_paths[i]  # 根据子图的索引选择对应的背景图
                    background_img = mpimg.imread(background_img_path)  # 读取背景图
                    # 显示背景图片，设置透明度
                    ax.imshow(background_img, extent=[0, 1920, 0, 1080], origin='upper', aspect='auto', alpha=1)

                    # 显示平滑后的热力图，调整alpha值
                    cax = ax.imshow(smoothed_heatmap.T, extent=[0, 1920, 0, 1080], origin='lower', aspect='auto', cmap='jet', alpha=0.5, norm=LogNorm())
                    ax.set_title(f'Image {i + 1}')
                    ax.set_xlabel('Gaze Position X')
                    ax.set_ylabel('Gaze Position Y')

                    start_idx = end_idx

                # 添加颜色条，设置对数范围
                fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='horizontal')

                # 强制更新图形
                self.canvas.figure = fig
                self.canvas.draw_idle()

                # 更新状态信息
                self.message_label.setText("热图分析成功")
            except Exception as e:
                self.message_label.setText(f"热图分析失败：{str(e)}")


    def emotion(self):
        if self.data is not None:
            try:
                self.toggle_plot_button.setVisible(True)
                self.toggle_plot_button.setEnabled(True)
                # 定义情绪区域的标签
                emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
                
                # 初始化一个字典，用于统计每个情绪区域的数量
                emotion_counts = {label: 0 for label in emotion_labels}
                
                # 获取眼动数据的 X 和 Y 坐标
                gaze_x = self.data['bino_eye_gaze_position_x']
                gaze_y = self.data['bino_eye_gaze_position_y']
                
                # 遍历数据并根据坐标判断属于哪个区域
                for x, y in zip(gaze_x, gaze_y):
                    for label, bounds in self.aoi_bounds.items():
                        x_min, x_max, y_min, y_max = bounds
                        if x_min <= x < x_max and y_min <= y < y_max:
                            emotion_counts[label] += 1
                            break  # 一旦找到对应区域就停止判断，避免多次统计
                
                # 计算每个情绪区域的占比
                total_data_points = len(self.data)
                emotion_percentages = {label: (count / total_data_points) * 100 for label, count in emotion_counts.items()}
                
                # 清除现有图形
                self.canvas.figure.clear()
                
                # 根据当前的图表类型绘制图表
                if self.plot_type == 'bar':
                    self.plot_bar_chart(emotion_percentages, emotion_labels)
                else:
                    self.plot_radar_chart(emotion_percentages, emotion_labels)
            
            except Exception as e:
                self.message_label.setText(f"情绪占比分析失败：{str(e)}")


    def plot_bar_chart(self, emotion_percentages, emotion_labels):
       
        # 创建条形图
        canvas_width = self.canvas.size().width()
        canvas_height = self.canvas.size().height()

        # 使用canvas的尺寸来设置figsize
        # 以英寸为单位，因此需要除以DPI来转换为英寸
        dpi = self.canvas.figure.dpi  # 获取当前figure的dpi
        figsize = (canvas_width / dpi, canvas_height / dpi)  # 计算figsize

        # 创建2x2的子图，使用适配画布的尺寸
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.plasma(np.linspace(0, 1, len(emotion_percentages)))
        bars=ax.bar(emotion_percentages.keys(), emotion_percentages.values(), color=colors, edgecolor='black', linewidth=1.2)
        # 设置X轴和Y轴标签
        ax.set_xlabel('AOI (Emotion)', fontsize=14, fontweight='bold', family='Arial')
        ax.set_ylabel('Proportion of Total Viewing Time (%)', fontsize=14, fontweight='bold', family='Arial')

        # 设置标题
        ax.set_title('Viewing Time Proportion per AOI', fontsize=16, fontweight='bold', family='Arial')
        max_value = max(emotion_percentages.values())
        ax.set_ylim(0, max(35, max_value + 10))  # Y轴最大值至少为100
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

        # 添加细致的网格线
        ax.grid(True, axis='y', linestyle='--', color='grey', alpha=0.3)
        
        # 强制更新图形
        self.canvas.figure = fig
        self.canvas.draw_idle()
        
        # 更新消息标签
        self.message_label.setText("情绪占比分析(条形图)")

    def plot_radar_chart(self, emotion_percentages, emotion_labels):
        # 雷达图数据准备
        values = [emotion_percentages[label] for label in emotion_labels]

        # 计算每个维度的角度
        angles = np.linspace(0, 2 * np.pi, len(emotion_labels), endpoint=False).tolist()

        # 将第一个值再加到最后面，闭合图形
        values += values[:1]
        angles += angles[:1]

        # 获取当前画布的尺寸，适配图形大小
        canvas_width = self.canvas.size().width()
        canvas_height = self.canvas.size().height()
        dpi = self.canvas.figure.dpi  # 获取当前figure的dpi
        figsize = (canvas_width / dpi, canvas_height / dpi)  # 计算figsize

        # 创建极坐标系的子图
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        # 设置雷达图的角度偏移和方向
        ax.set_theta_offset(np.pi / 2)  # 调整雷达图的起始角度
        ax.set_theta_direction(-1)  # 逆时针方向

        # 绘制雷达图的边框和填充
        ax.plot(angles, values, linewidth=2, linestyle='solid', label='Emotion Proportions', color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')

        # 设置极径的标签为空
        ax.set_yticklabels([])

        # 设置每个角的标签（情绪区域名称）
        ax.set_xticks(angles[:-1])  # 去掉最后一个重复的角度
        ax.set_xticklabels(emotion_labels, fontsize=14, fontweight='bold', color='black', rotation=45, ha='right')

        # 添加数据注释：在雷达图上显示每个情绪区域的百分比
        for i in range(len(emotion_labels)):
            ax.text(angles[i], values[i] + 2, f'{values[i]:.1f}%', horizontalalignment='center', size=12, color='black')

        # 设置雷达图标题，并调整位置
        ax.set_title('Emotion Proportions Radar', size=16, fontweight='bold', color='black', pad=20)

        # 增加顶部边距，避免标题和标签重叠
        plt.subplots_adjust(top=0.9)

        # 添加网格线，细致的网格线
        ax.grid(True, axis='both', linestyle='--', color='grey', alpha=0.3)

        # 强制更新图形
        self.canvas.figure = fig
        self.canvas.draw_idle()  # 强制更新图形
        # 更新消息标签
        self.message_label.setText("情绪占比分析(雷达图)")

       
    def show_statistics(self):
        if self.data is not None:
            try:
                # 采样率为200Hz
                sampling_rate = 200  # Hz
                fixation_duration_threshold = 20  # 100ms， 即200Hz下为40个数据点
                saccade_distance_threshold = 50  # 扫视阈值，眼睛移动超过50像素视为扫视

                trigger_index = self.data[self.data['trigger'] != 0].index.min()
                    
                if pd.isna(trigger_index):
                    self.message_label.setText("<font color='red' style='font-size: 14px;'><b>未找到有效的trigger数据</b></font>")
                    return
                    
                # 计算每次眼动的距离，单位为像素
                self.data['distance'] = np.sqrt(
                    (self.data['bino_eye_gaze_position_x'].diff())**2 +
                    (self.data['bino_eye_gaze_position_y'].diff())**2
                ).fillna(0)
                    
                # 从第一个触发器开始分析数据
                self.data = self.data.loc[trigger_index:].reset_index(drop=True)    
                # 计算眼动数据的时间差，单位为秒
                self.data['time_diff'] = self.data['timestamp'].diff().fillna(0)

                # 定义情绪区域的标签
                emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
                    
                # 初始化一个字典，用于统计每个情绪区域的数量
                emotion_counts = {label: 0 for label in emotion_labels}

                # 获取眼动数据的 X 和 Y 坐标
                gaze_x = self.data['bino_eye_gaze_position_x']
                gaze_y = self.data['bino_eye_gaze_position_y']
                
                # 瞳孔平均直径大小
                self.data['average_pupil_diameter_mm'] = (self.data['left_eye_pupil_diameter_mm'] + self.data['right_eye_pupil_diameter_mm']) / 2
                average_pupil_diameter = np.mean(self.data['average_pupil_diameter_mm'])
                
                # 计算瞳孔直径与平均瞳孔直径之间的差异
                self.data['left_pupil_diff'] = np.abs(self.data['left_eye_pupil_diameter_mm'] - self.data['average_pupil_diameter_mm'])
                self.data['right_pupil_diff'] = np.abs(self.data['right_eye_pupil_diameter_mm'] - self.data['average_pupil_diameter_mm'])

                # 计算差异的标准差，表示个体相对于自己平均瞳孔直径的变化大小
                std_diff_left = np.std(self.data['left_pupil_diff'])
                std_diff_right = np.std(self.data['right_pupil_diff'])
                cv_left = std_diff_left / np.mean(self.data['left_eye_pupil_diameter_mm'])
                cv_right = std_diff_right / np.mean(self.data['right_eye_pupil_diameter_mm'])
                
                
                # 计算平均的变化大小
                average_std_diff = np.mean([std_diff_left, std_diff_right])
                
                
                
                # 遍历数据并根据坐标判断属于哪个区域
                for x, y in zip(gaze_x, gaze_y):
                    for label, bounds in self.aoi_bounds.items():
                        x_min, x_max, y_min, y_max = bounds
                        if x_min <= x < x_max and y_min <= y < y_max:
                            emotion_counts[label] += 1
                            break  # 一旦找到对应区域就停止判断，避免多次统计

                # 计算每个情绪区域的占比
                total_counts = len(self.data)
                emotion_percentages = {label: (count / total_counts) * 100 for label, count in emotion_counts.items()}

                # 计算静态注视熵
                def calculate_entropy(percentages):
                    probabilities = percentages / 100  # 转换为概率
                    entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))  # 加上微小值避免log(0)
                    return entropy

                static_entropy = calculate_entropy(np.array(list(emotion_percentages.values())))

                # 计算眼跳注视熵
                def calculate_transition_entropy(aoi_counts, steady_state_distribution):
                    # 创建空的转移计数矩阵
                    transition_counts = np.zeros((len(aoi_counts), len(aoi_counts)))

                    # 统计从一个AOI到另一个AOI的转移次数
                    total_transitions = 0
                    prev_aoi = None  # 初始化前一个AOI
                    for idx, row in self.data.iterrows():
                        # 获取当前的AOI
                        current_aoi = None
                        x, y = row['bino_eye_gaze_position_x'], row['bino_eye_gaze_position_y']
                        
                        # 判断当前坐标属于哪个AOI
                        for label, bounds in self.aoi_bounds.items():
                            x_min, x_max, y_min, y_max = bounds
                            if x_min <= x < x_max and y_min <= y < y_max:
                                current_aoi = label
                                break
                        
                        if current_aoi and prev_aoi and current_aoi != prev_aoi:  # 当前AOI与上一个AOI不同，记录转移
                            transition_counts[emotion_labels.index(prev_aoi), emotion_labels.index(current_aoi)] += 1
                            total_transitions += 1
                        
                        prev_aoi = current_aoi  # 更新前一个AOI

                    # 计算转移概率
                    transition_probabilities = transition_counts / total_transitions
                    entropy = 0

                    # 计算条件熵
                    for i in range(len(aoi_counts)):
                        row_entropy = 0
                        for j in range(len(aoi_counts)):
                            if transition_probabilities[i, j] > 0:
                                row_entropy -= transition_probabilities[i, j] * np.log2(transition_probabilities[i, j] + np.finfo(float).eps)
                        # 乘以该行的平稳分布
                        entropy += steady_state_distribution[i] * row_entropy
                    return entropy

                transition_entropy = calculate_transition_entropy(emotion_counts, np.array(list(emotion_percentages.values())) )

                # 计算注视次数：根据时间差和移动距离
                fixations = 0
                fixation_counter = 0  # 用于计算注视的持续时间（以数据点为单位）
                for idx, row in self.data.iterrows():
                    if row['distance'] < saccade_distance_threshold:  # 移动距离小，认为是注视
                        fixation_counter += 1
                    else:  # 移动距离大，认为是扫视
                        if fixation_counter >= fixation_duration_threshold:
                            fixations += 1
                        fixation_counter = 0  # 重置注视计数器

                # 如果最后一个注视没有被计入，手动计入
                if fixation_counter >= fixation_duration_threshold:
                    fixations += 1

                # 计算扫视次数：根据时间差和移动距离
                saccades = 0
                for idx, row in self.data.iterrows():
                    if row['distance'] > saccade_distance_threshold:  # 移动距离大，认为是扫视
                        saccades += 1


                



            
             # 生成优化后的统计字符串
                statistics_str = (
                    f"<div style='font-family: Arial, sans-serif; color: #333; font-size: 16px; line-height: 1.4;'>"
                    f"<b><font size='5' color='#4A90E2'>眨眼次数:</font></b> <font size='5' color='#333333'>{self.blink_count}</font><br>"

                    f"<b><font size='5' color='#4A90E2'>注视次数:</font></b> <font size='5' color='#333333'>{fixations}</font><br>"

                    f"<b><font size='5' color='#4A90E2'>扫视次数:</font></b> <font size='5' color='#333333'>{saccades}</font><br>"

                    f"<b><font size='5' color='#7ED321'>平均瞳孔直径 (mm):</font></b> <font size='5' color='#333333'>{average_pupil_diameter:.2f}</font><br>"

                    f"<b><font size='5' color='#9B51E0'>静态注视熵:</font></b> <font size='5' color='#333333'>{static_entropy:.4f}</font><br>"

                    f"<font size='4' color='gray'>参考均匀分布的注视熵：2.58</font><br>"

                    f"<b><font size='5' color='#F39C12'>眼跳注视熵:</font></b> <font size='5' color='#333333'>{transition_entropy:.4f}</font><br>"

                    f"<b><font size='5' color='#9B51E0'>左眼瞳孔变化变异系数:</font></b> <font size='5' color='#333333'>{cv_left:.4f}</font><br>"
                    f"<b><font size='5' color='#F39C12'>右眼瞳孔变化变异系数:</font></b> <font size='5' color='#333333'>{cv_right:.4f}</font><br>"
                    f"<b><font size='5' color='#32CD32'>瞳孔直径变化标准差:</font></b> <font size='5' color='#333333'>{average_std_diff:.2f}</font><br>"

                    f"<font size='4' color='gray' style='font-size: 14px;'>统计完毕，数据已更新。</font>"
                    f"</div>"
                )




                self.canvas.figure.clf()
                self.canvas.draw_idle()
                # 设置更新消息
                self.message_label.setText(statistics_str)

            except Exception as e:
                self.message_label.setText(f"<font color='red' style='font-size: 14px;'><b>统计量计算失败：</b>{str(e)}</font>")




    def toggle_plot(self):
            if self.plot_type == 'bar':
                self.plot_type = 'radar'
                self.toggle_plot_button.setText("切换到条形图")
            else:
                self.plot_type = 'bar'
                self.toggle_plot_button.setText("切换到雷达图")

            # 重新绘制图表
            self.emotion()  # 重新绘制情绪占比图


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CsvVisualizerApp()
    window.show()
    sys.exit(app.exec_())
