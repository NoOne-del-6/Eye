import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton

class HeatmapWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题
        self.setWindowTitle('左右切换图片')
        
        # 创建垂直布局
        self.layout = QVBoxLayout()
        
        # 创建标签来显示图片
        self.label = QLabel(self)
        self.layout.addWidget(self.label)
        
        # 创建上一张和下一张按钮
        self.prev_button = QPushButton('上一张', self)
        self.next_button = QPushButton('下一张', self)
        self.layout.addWidget(self.prev_button)
        self.layout.addWidget(self.next_button)
        
        # 设置窗口布局
        self.setLayout(self.layout)
        
        # 假设有16张图片文件
        self.image_paths = [f"image{i}.png" for i in range(1, 17)]
        
        # 当前显示的图片索引
        self.current_index = 0
        
        # 连接按钮的点击事件
        self.prev_button.clicked.connect(self.show_prev_image)
        self.next_button.clicked.connect(self.show_next_image)
        
        # 显示第一张图片
        self.show_image()

    def show_image(self):
        # 加载并显示当前图片
        pixmap = QPixmap(self.image_paths[self.current_index])
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignCenter)

    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            # 显示下一张图片
            self.current_index += 1
            self.show_image()

    def show_prev_image(self):
        if self.current_index > 0:
            # 显示上一张图片
            self.current_index -= 1
            self.show_image()

# 创建应用程序实例
app = QApplication(sys.argv)

# 创建并显示窗口
window = HeatmapWindow()
window.resize(800, 600)
window.show()

# 运行应用程序的主事件循环
sys.exit(app.exec_())
