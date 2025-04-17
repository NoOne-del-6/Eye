import sys
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow,QTableWidget, QTableWidgetItem,QScrollArea,QHBoxLayout, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QSizePolicy, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
from matplotlib.offsetbox import AnchoredText
import os
from natsort import natsorted
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier




class DataProcess:
    def __init__(self, data=None):
        self.data = data
       
    def load_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "选择 CSV 文件", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                return True, self.data
            except Exception as e:
                return False, str(e)
        return False, "No file selected"
    
    def remove_invalid_data(self):
        if self.data is not None:
            try:
                original_length = len(self.data)
                self.data = self.data[((self.data['left_eye_valid'] != 0) & (self.data['right_eye_valid'] != 0) & (self.data['bino_eye_valid'] != 0)) | (self.data['trigger'] != 0)]
                removed_length = original_length - len(self.data)
                return True, removed_length
            except Exception as e:
                return False, f"Data processing failed: {str(e)}"
        return False, "No data to process"
    
    def calculate_blink_count(self):
        if self.data is not None:
            blink_count = 0
            in_blink = False
            for i in range(1, len(self.data)):
                if self.data['left_eye_valid'][i] == 0 and self.data['left_eye_valid'][i - 1] != 0:
                    in_blink = True
                elif self.data['left_eye_valid'][i] != 0 and in_blink:
                    blink_count += 1
                    in_blink = False
            return blink_count
        return 0

class ImageSelector:
    def __init__(self):
        self.image_paths = []

    def select_images(self):
        options = QFileDialog.Options()
        folder = QFileDialog.getExistingDirectory(None, "选择包含图片的文件夹", options=options)
        if folder:
            self.image_paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if self.image_paths:
                return True, self.image_paths
            return False, "No valid images found in the selected folder"
        return False, "No folder selected"

    def get_selected_images(self):
        return self.image_paths

class PlotShow:
    def __init__(self, canvas):
        self.canvas = canvas
        self.current_heatmap_index = 1

    def clear_canvas(self):
        self.canvas.figure.clear()

    def resize_canvas(self):
        canvas_width = self.canvas.size().width()
        canvas_height = self.canvas.size().height()
        dpi = self.canvas.figure.dpi
        figsize = (canvas_width / dpi, canvas_height / dpi)
        self.canvas.figure.set_size_inches(figsize)
        self.canvas.draw_idle()

    def plot_pupil(self, data, features, title, ylabel, feature_labels):
        fig = self.canvas.figure
        ax = fig.add_subplot(111)
        ax.clear()
        # 如果数据中没有时间列，生成一个时间列作为 x 轴
        if 'time' not in data.columns:
            data['time'] = np.arange(len(data)) / 200  # 假设采样率为 200Hz
        # 绘制每个特征的曲线
        for feature, label in zip(features, feature_labels):
            ax.plot(
                data['time'],  # 使用时间作为 x 轴
                data[feature],
                label=label,
                alpha=0.7
            )

        # 在触发点为202的位置绘制垂直分割线
        trigger_indices = data.index[data['trigger'] == 202].tolist()
        for idx in trigger_indices:
            time_value = data.loc[idx, 'time']  # 获取对应的时间值
            ax.axvline(x=time_value, color='red', linestyle='--', alpha=0.8)

        # 设置 x 轴刻度为每 5 秒一个分割
        time_min = data['time'].min()
        time_max = data['time'].max()
        ax.set_xticks(np.arange(time_min, time_max + 1, 5))  # 每 5 秒一个刻度
        ax.set_xticklabels([f"{t:.1f}s" for t in np.arange(time_min, time_max + 1, 5)], fontsize=10)

        # 设置图表标题和标签
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

        # 调整图像尺寸和更新画布
        self.resize_canvas()
        self.canvas.draw_idle()

    def plot_heatmap(self, data, background_image_paths):
        # 获取触发点的索引，仅保留值为202的触发点
        trigger_indices = data.index[data['trigger'] == 202].tolist()
        if not trigger_indices:
            return "未找到有效的触发数据"
       
        # 确保触发点数量与背景图片数量一致
        if len(trigger_indices) > len(background_image_paths):
            return "图片数量缺少"
        
        # 存储热图和背景图片，用于导航
        self.heatmaps = []
        self.background_images = []
        
        for i in range(len(trigger_indices)):
            if i < len(trigger_indices) - 1:
                # 如果不是最后一个触发点，结束位置是下一个触发点
                start_idx = trigger_indices[i]
                end_idx = trigger_indices[i + 1]
            else:
                start_idx = trigger_indices[i]
                end_idx = data.index[-1]  # 获取数据的最后一行的索引

            # 获取当前段的数据
            segment_data = data.iloc[start_idx:end_idx]

            # 计算二维直方图，生成热图数据
            heatmap_data, xedges, yedges = np.histogram2d(
                segment_data['bino_eye_gaze_position_x'],  # 眼动数据的X坐标
                segment_data['bino_eye_gaze_position_y'],  # 眼动数据的Y坐标
                bins=[50, 28],  # 热图的分辨率（X方向50格，Y方向28格）
                range=[[0, 1920], [0, 1080]]  # 坐标范围
            )

            # 对热图数据进行高斯平滑处理
            smoothed_heatmap = gaussian_filter(heatmap_data, sigma=0.8)
            self.heatmaps.append(smoothed_heatmap)  # 保存平滑后的热图数据
            self.background_images.append(mpimg.imread(background_image_paths[i]))  # 保存对应的背景图片
            start_idx = end_idx
        self.display_current_heatmap()  # 显示当前热图

    def display_current_heatmap(self):
        # 检查是否存在热图数据，如果没有则直接返回
        if not hasattr(self, 'heatmaps') or not self.heatmaps:
            return
        # 清空画布并绘制当前热图
        self.clear_canvas()
        fig = self.canvas.figure
        ax = fig.add_subplot(111)
        ax.clear()

        # 获取当前热图和背景图
        current_heatmap = self.heatmaps[self.current_heatmap_index - 1]
        background_img = self.background_images[self.current_heatmap_index - 1]

        # 绘制背景图
        ax.imshow(background_img, extent=[0, 1920, 0, 1080], origin='upper', aspect='auto', alpha=1)
        # 绘制热图，使用对数归一化(LogNorm)来增强对比度
        cax = ax.imshow(current_heatmap.T, extent=[0, 1920, 0, 1080], origin='upper', aspect='auto', cmap='jet', alpha=0.5, norm=LogNorm())
        # 设置标题和坐标轴标签
        ax.set_title(f'Image {self.current_heatmap_index}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Gaze Position X', fontsize=12)
        ax.set_ylabel('Gaze Position Y', fontsize=12)

        # 添加热图的颜色条
        cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Heatmap Intensity', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # 添加当前热图索引的注释
        anchored_text = AnchoredText(f"Heatmap {self.current_heatmap_index}/{len(self.heatmaps)}",
                                     loc='upper right', prop=dict(size=10, weight='bold'), frameon=True)
        ax.add_artist(anchored_text)

        # 调整画布大小并刷新
        self.resize_canvas()
        self.canvas.draw_idle()

    def previous_heatmap(self):
        # 如果有热图并且当前不是第一张，则显示上一张热图
        if hasattr(self, 'heatmaps') and self.current_heatmap_index > 1:
            self.current_heatmap_index -= 1
            self.display_current_heatmap()

    def next_heatmap(self):
        # 如果有热图并且当前不是最后一张，则显示下一张热图
        if hasattr(self, 'heatmaps') and self.current_heatmap_index < len(self.heatmaps):
            self.current_heatmap_index += 1
            self.display_current_heatmap()

    def plot_chart(self, emotion_percentages, emotion_labels):
        self.clear_canvas()  
        fig = self.canvas.figure
        ax = fig.add_subplot(111)
        ax.clear()
        colors = plt.cm.plasma(np.linspace(0, 1, len(emotion_labels)))
        bars = ax.bar(emotion_labels, [emotion_percentages[label] for label in emotion_labels], 
                      color=colors, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('AOI (Emotion)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Proportion of Total Viewing Time (%)', fontsize=14, fontweight='bold')
        ax.set_title('Viewing Time Proportion per AOI', fontsize=16, fontweight='bold')
        max_value = max(emotion_percentages.values())
        ax.set_ylim(0, max(35, max_value + 10))
        for bar, label in zip(bars, emotion_labels):
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', color='grey', alpha=0.3)

        self.resize_canvas()
        self.canvas.draw_idle()

    def plot_statistics(self, statistics_str):
        # 清空画布
        self.clear_canvas()

        # 创建一个新的 Figure 和 Axes
        fig = self.canvas.figure
        ax = fig.add_subplot(111)
        ax.clear()

        # 将统计字符串分行解析为键值对
        lines = statistics_str.strip().split("\n")
        data = [line.split(":\t") for line in lines if ":\t" in line]

        # 创建表格数据
        table_data = [[key, value] for key, value in data]

        # 隐藏坐标轴
        ax.axis('tight')
        ax.axis('off')

        # 创建表格
        table = ax.table(cellText=table_data, colLabels=["指标", "值"], loc='center', cellLoc='center')

        # 设置表头字体颜色为红色且加粗，并增大字体
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # 表头行
                cell.set_text_props(color='red', fontweight='bold', fontsize=14)

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)  # 增大字体大小
        table.auto_set_column_width(col=list(range(len(table_data[0]))))

        # 调整表格字体以支持中文显示
        rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体为 SimHei 支持中文
        rcParams['axes.unicode_minus'] = False  # 避免负号显示问题

        # 调整表格位置和大小以适应画布
        table.scale(2.5, 2.5)  # 调整表格大小，使其更大

        # 调整画布边距以减少空白
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        # 调整画布大小并刷新
        self.resize_canvas()
        self.canvas.draw_idle()

class AnalyzeData:
    
    def __init__(self, data):
        self.data = data
        self.aoi_bounds = {
            "A_top_left": [0, 640, 0, 540],        # 左上
            "B_top_middle": [640, 1280, 0, 540],   # 中上
            "C_top_right": [1280, 1920, 0, 540],   # 右上
            "A_bottom_left": [0, 640, 540, 1080],  # 左下
            "B_bottom_middle": [640, 1280, 540, 1080],  # 中下
            "C_bottom_right": [1280, 1920, 540, 1080]  # 右下
        }
        
        # 每个区域对应的情绪标签（这里是区域到标签的映射）
        self.region_to_emotion = {
            "A_top_left": "Positive",
            "B_top_middle": "Neutral",
            "C_top_right": "Negative",
            "A_bottom_left": "Positive",
            "B_bottom_middle": "Neutral",
            "C_bottom_right": "Negative"
        }

        # 情绪标签
        self.emotion_labels = ['Positive', 'Neutral', 'Negative']
        
    # 计算连续眼动点之间的距离
    def calculate_distance(self):
        gaze_x = self.data['bino_eye_gaze_position_x']
        gaze_y = self.data['bino_eye_gaze_position_y']
        self.data['distance'] = np.sqrt((gaze_x.diff())**2 + (gaze_y.diff())**2).fillna(0)

    # 计算注视次数
    def calculate_fixations(self, saccade_distance_threshold, fixation_duration_threshold):
        fixations = 0
        fixation_counter = 0  # 记录注视的持续时间（以数据点为单位）
        for idx, row in self.data.iterrows():
            if row['distance'] < saccade_distance_threshold:  # 如果距离小，认为是注视
                fixation_counter += 1
            else:  # 如果距离大，认为是扫视
                if fixation_counter >= fixation_duration_threshold:
                    fixations += 1
                fixation_counter = 0  # 重置注视计数器
        if fixation_counter >= fixation_duration_threshold:
            fixations += 1
        return fixations

    #  计算扫视次数  
    def calculate_saccades(self, saccade_distance_threshold):
        return sum(self.data['distance'] > saccade_distance_threshold)

    # 计算每个情绪区域的占比
    def calculate_emotion_percentages(self):
        if self.data is None or self.data.empty:
            raise ValueError("数据为空，无法计算情绪占比")
        
        # 初始化情绪计数字典，键为情绪标签
        emotion_counts = {emotion: 0 for emotion in self.emotion_labels}  # 使用情绪标签作为字典的键
        
        # 获取眼动数据的 X 和 Y 坐标
        gaze_x = self.data['bino_eye_gaze_position_x']
        gaze_y = self.data['bino_eye_gaze_position_y']
        
        # 遍历眼动数据
        for x, y in zip(gaze_x, gaze_y):
            for label, bounds in self.aoi_bounds.items():
                x_min, x_max, y_min, y_max = bounds
                # 检查眼动数据是否在对应区域内
                if x_min <= x < x_max and y_min <= y < y_max:
                    # 获取该区域的情绪标签
                    emotion_label = self.region_to_emotion[label]
                    # 更新情绪计数
                    emotion_counts[emotion_label] += 1
                    break  # 一旦找到匹配的AOI，就停止
        
        total_data_points = len(self.data)
        if total_data_points == 0:
            raise ValueError("数据点数量为零，无法计算情绪占比")
        
        # 计算每个情绪标签的占比
        emotion_percentages = {emotion: (count / total_data_points) * 100
                               for emotion, count in emotion_counts.items()}
        return emotion_percentages
    
    # 计算静态注视熵
    def calculate_static_entropy(self, emotion_percentages):
        probabilities = np.array(list(emotion_percentages.values())) / 100  # 转换为概率
        probabilities = probabilities[probabilities > 0]  # 去除为零的概率值，避免log(0)
        entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))  # 计算熵
        return entropy

    # 计算眼跳注视熵
    def calculate_transition_entropy(self, emotion_percentages):
        # 创建空的转移计数矩阵
        num_emotions = len(self.emotion_labels)
        transition_counts = np.zeros((num_emotions, num_emotions))

        # 统计从一个情绪区域到另一个情绪区域的转移次数
        total_transitions = 0
        prev_emotion = None  # 初始化前一个情绪
        for idx, row in self.data.iterrows():
            # 获取当前的情绪区域
            current_emotion = None
            x, y = row['bino_eye_gaze_position_x'], row['bino_eye_gaze_position_y']

            # 判断当前坐标属于哪个情绪区域
            for label, bounds in self.aoi_bounds.items():
                x_min, x_max, y_min, y_max = bounds
                if x_min <= x < x_max and y_min <= y < y_max:
                    current_emotion = self.region_to_emotion[label]
                    break

            if current_emotion and prev_emotion and current_emotion != prev_emotion:  # 当前情绪与上一个情绪不同，记录转移
                transition_counts[self.emotion_labels.index(prev_emotion), self.emotion_labels.index(current_emotion)] += 1
                total_transitions += 1

            prev_emotion = current_emotion  # 更新前一个情绪

        if total_transitions == 0:
            return 0  # 如果没有转移，返回熵为0

        # 计算转移概率
        transition_probabilities = transition_counts / total_transitions
        entropy = 0

        # 计算条件熵
        for i in range(num_emotions):
            row_entropy = 0
            for j in range(num_emotions):
                if transition_probabilities[i, j] > 0:
                    row_entropy -= transition_probabilities[i, j] * np.log2(transition_probabilities[i, j])
            # 乘以该行的平稳分布
            entropy += (emotion_percentages[self.emotion_labels[i]] ) * row_entropy
        return entropy

    def calculate_statistics(self, blink_count):
        # 采样率与阈值设置
        sampling_rate = 200
        fixation_duration_threshold = 20  # 100ms， 即200Hz下为20个数据点
        saccade_distance_threshold = 50  # 扫视阈值，眼睛移动超过50像素视为扫视

        # 计算眼动数据
        self.calculate_distance()

        # 计算注视次数与扫视次数
        fixations = self.calculate_fixations(saccade_distance_threshold, fixation_duration_threshold)
        saccades = self.calculate_saccades(saccade_distance_threshold)

        # 计算情绪占比
        emotion_percentages = self.calculate_emotion_percentages()

        # 计算静态注视熵
        static_entropy = self.calculate_static_entropy(emotion_percentages)

        # 计算眼跳注视熵
        transition_entropy = self.calculate_transition_entropy(emotion_percentages)

        # 计算左眼和右眼瞳孔直径的标准差
        std_diff_left = self.data['left_eye_pupil_diameter_mm'].std()
        std_diff_right = self.data['right_eye_pupil_diameter_mm'].std()

        # 生成统计字符串，适合表格形式
        statistics_str = (
            "眨眼次数:\t{}\n"
            "注视次数:\t{}\n"
            "扫视次数:\t{}\n"
            "静态注视熵(参考 2.58):\t{:.4f}\n"
            "眼跳注视熵:\t{:.4f}\n"
            "左眼瞳标准差:\t{:.4f}\n"
            "右眼瞳标准差:\t{:.4f}\n"
            "统计完毕，数据已更新。\n"
        ).format(
            blink_count, fixations, saccades, static_entropy, 
            transition_entropy, std_diff_left, std_diff_right
        )
        return statistics_str
        
class EmotionPrediction:
    def __init__(self, features):
        self.features = features
        # 初始化模型配置
        self.models = self.initialize_models()

    def initialize_models(self):
        # 这些模型参数是已经训练好的，直接加载训练好的模型
        models = {
            'RF': RandomForestClassifier(max_depth=5, max_features='sqrt', min_samples_leaf=1, min_samples_split=5, n_estimators=100, random_state=42),
            'SVC': SVC(C=0.1, kernel='linear', gamma='scale'),
            'Logistic Regression': LogisticRegression(C=0.1, solver='saga'),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8)
        }
        return models

    def preprocess_features(self, features):
        # 提取数值特征，用于预测
        feature_data = [
            features['blink_count'], 
            features['fixations'], 
            features['saccades'],
            features['static_entropy'],
            features['transition_entropy'],
            features['std_diff_left'],
            features['std_diff_right'],
            features['Positive'],
            features['Neutral'],
            features['Negative']
        ]
        
        # 将特征转换为适合模型的格式
        return np.array(feature_data).reshape(1, -1)

    def predict(self):
        # 预处理特征
        X_new = self.preprocess_features(self.features)

        # 标准化特征数据
        scaler = RobustScaler()
        X_new_scaled = scaler.fit_transform(X_new)

        # 存储所有模型的预测结果
        predictions = {}

        # 预测每个模型的情绪标签
        for model_name, model in self.models.items():
            # 使用模型进行预测
            prediction = model.predict(X_new_scaled)  # 获取预测结果
            
            # 根据预测的类别标签返回情绪
            predicted_label = self.get_emotion_label(prediction[0])  # 获取预测的情绪标签
            predictions[model_name] = predicted_label  # 保存预测结果
            print(f"{model_name} 情绪预测结果: {predicted_label}")  # 显示预测结果

        return predictions

    def get_emotion_label(self, predicted_class):
        # 根据预测的数字标签返回情绪标签
        # 0 -> negative
        # 1 -> neutral
        # 2 -> positive
        if predicted_class == 0:
            return 'negative'  # 负面情绪
        elif predicted_class == 1:
            return 'neutral'   # 中性情绪
        elif predicted_class == 2:
            return 'positive'  # 正面情绪
        else:
            return 'unknown'   # 如果有未知类别（例如模型可能返回4等）
        
    
    
class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(r'C:\Users\Lhtooo\Desktop\生理信号分析\code\images\bg.jpg'))  # 设置窗口图标
        self.setWindowTitle("眼动工具箱")
        self.setGeometry(300, 150, 1100, 800)
        # 初始化数据处理、绘图、数据分析类
        self.data_process = DataProcess()
        self.plotter = PlotShow(FigureCanvas(Figure(figsize=(10,8))))
        self.blink_count = 0
        self.features = {}
        self.initUI()
        
    def initUI(self):
    
        # 创建中央部件
        central_widget = QWidget(self)
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
        
        # 添加绘图按钮
        button_layout = QHBoxLayout()
        self.pupil_button = QPushButton('瞳孔可视化')
        self.pupil_button.clicked.connect(self.pupil_plot)
        self.pupil_button.setEnabled(False)  # 初始状态不可用
        button_layout.addWidget(self.pupil_button)
        
        self.heatmap_button = QPushButton('热图分析')
        self.heatmap_button.clicked.connect(self.heatmap)
        self.heatmap_button.setEnabled(False)  # 初始状态不可用
        button_layout.addWidget(self.heatmap_button)
        
        self.emotion_button = QPushButton('情绪占比')
        self.emotion_button.clicked.connect(self.emotion)
        self.emotion_button.setEnabled(False)  # 初始状态不可用
        button_layout.addWidget(self.emotion_button)
        
        self.stats_button = QPushButton('统计量')
        self.stats_button.clicked.connect(self.statistics)
        self.stats_button.setEnabled(False)  # 初始状态不可用
        button_layout.addWidget(self.stats_button)
        
        layout.addLayout(button_layout)
        
        self.predict_button = QPushButton('情绪分析')
        self.predict_button.clicked.connect(self.predict)
        self.predict_button.setEnabled(False)  # 初始状态不可用
        layout.addWidget(self.predict_button)
        
        
        # 将画布添加到布局中
        self.canvas = self.plotter.canvas
        layout.addWidget(self.canvas)
        
        
        # 添加热图导航按钮
        heatmap_navigation_layout = QHBoxLayout()
        self.prev_button = QPushButton('上一张')
        self.prev_button.clicked.connect(self.prev_heatmap)
        self.prev_button.setFixedWidth(80)  # 设置按钮宽度
        self.prev_button.setEnabled(False)  # 初始状态不可用
        self.prev_button.setVisible(False)
        heatmap_navigation_layout.addWidget(self.prev_button)

        self.next_button = QPushButton('下一张')
        self.next_button.clicked.connect(self.next_heatmap)
        self.next_button.setFixedWidth(80)  # 设置按钮宽度
        self.next_button.setEnabled(False)  # 初始状态不可用
        self.next_button.setVisible(False)
        heatmap_navigation_layout.addWidget(self.next_button)

        layout.addLayout(heatmap_navigation_layout)
        
        
        # 添加显示消息的标签
        self.message_label = QLabel('请选择 眼动 文件进行加载')
        self.message_label.setAlignment(Qt.AlignCenter)  # 使文字居中
        self.message_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # 设置高度适应内容
        layout.addWidget(self.message_label)
        
        # 设置中央部件的布局
        central_widget.setLayout(layout)

        # 设置QMainWindow的中央部件
        self.setCentralWidget(central_widget)
        
    def load_csv(self):
        success, message = self.data_process.load_csv()
        self.analyze = AnalyzeData(self.data_process.data)
        if success:
            self.message_label.setText(f"数据加载成功！")
            self.blink_count = self.data_process.calculate_blink_count()
            self.process_button.setEnabled(True)
        else:
            self.message_label.setText(f"加载失败: {message}")
    
    def process_data(self):
        success, removed_length = self.data_process.remove_invalid_data()
        if success:
            self.message_label.setText(f"数据处理完成, 去除无效数据 {removed_length} 条")
            self.pupil_button.setEnabled(True)
            self.heatmap_button.setEnabled(True)
            self.emotion_button.setEnabled(True)
            self.stats_button.setEnabled(True)
            self.predict_button.setEnabled(True)
        else:
            self.message_label.setText(f"数据处理失败")
     
    def pupil_plot(self):
        self.plotter.clear_canvas()
        features = ['left_eye_pupil_diameter_mm', 'right_eye_pupil_diameter_mm']
        
        self.plotter.plot_pupil(
            self.data_process.data, 
            features, 
            'Pupil Diameter (Left and Right) Over Time', 
            'Pupil Diameter (mm)', 
            ['Left Pupil', 'Right Pupil']
        )
   
    def heatmap(self):
        self.plotter.clear_canvas()
        image_selector = ImageSelector()
        success, result = image_selector.select_images()
        if success:
            # 按文件名自然排序图片路径
            background_image_paths = natsorted(result, key=lambda x: os.path.basename(x))
            self.plotter.plot_heatmap(
                self.data_process.data,
                background_image_paths,
            )
          
            self.prev_button.setVisible(True)
            self.next_button.setVisible(True)
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
            # 更新消息标签
            self.message_label.setText("热图分析成功")
        else:
            # 更新消息标签显示错误信息
            self.message_label.setText(f"热图分析失败: {result}")
       
    def prev_heatmap(self):
        try:
            self.plotter.previous_heatmap()
            self.message_label.setText(f"显示上一张热图 (当前第 {self.plotter.current_heatmap_index} 张)")
        except Exception as e:
            self.message_label.setText(f"无法显示上一张热图: {str(e)}")

    def next_heatmap(self):
        try:
            self.plotter.next_heatmap()
            self.message_label.setText(f"显示下一张热图 (当前第 {self.plotter.current_heatmap_index} 张)")
        except Exception as e:
            self.message_label.setText(f"无法显示下一张热图: {str(e)}")
    
    def emotion(self):
        # 计算情绪占比
        emotion_percentages = self.analyze.calculate_emotion_percentages()
        # 清除画布并绘制条形图
        self.plotter.clear_canvas()
        self.plotter.plot_chart(emotion_percentages, self.analyze.emotion_labels)
        # 更新消息标签
        self.message_label.setText("情绪占比分析成功")

    def statistics(self):
        try:
            # 调用统计方法并获取统计字符串
            statistics_str = self.analyze.calculate_statistics(self.blink_count)
            # 在画布上显示统计信息
            self.plotter.plot_statistics(statistics_str)
            # 更新消息标签
            self.message_label.setText("统计计算成功")
        except Exception as e:
            self.message_label.setText(f"统计计算失败: {str(e)}")

    def predict(self):
        try:
            # 进行情绪预测
            
            emotion_percentages = self.analyze.calculate_emotion_percentages()
            statistics_str = self.analyze.calculate_statistics(self.blink_count)
             # 将情绪占比拆分并添加到 features 中
            for emotion, percentage in emotion_percentages.items():
                self.features[f'{emotion}'] = percentage / 100
            # 将 statistics_str 拆分并添加到 features 中
            # 提取统计字符串中的数值部分
            statistics_data = {
                'blink_count': self.blink_count,
                'fixations': int(statistics_str.split('\n')[1].split(':')[1].strip()),
                'saccades': int(statistics_str.split('\n')[2].split(':')[1].strip()),
                'static_entropy': float(statistics_str.split('\n')[3].split(':')[1].strip()),
                'transition_entropy': float(statistics_str.split('\n')[4].split(':')[1].strip()),
                'std_diff_left': float(statistics_str.split('\n')[5].split(':')[1].strip()),
                'std_diff_right': float(statistics_str.split('\n')[6].split(':')[1].strip())
            }
            # 将提取的统计数据添加到 features 中
            for key, value in statistics_data.items():
                self.features[key] = value

            emotion_predictor = EmotionPrediction(self.features)
            predictions = emotion_predictor.predict()
            # 打印所有模型的预测结果
            print(predictions)
            self.message_label.setText(f"情绪预测完成")
        except Exception as e:
            self.message_label.setText(f"情绪预测失败: {str(e)}")
      


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
