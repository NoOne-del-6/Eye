from PyQt5.QtWidgets import QHBoxLayout,QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QSpacerItem, QSizePolicy
import sys

from PyQt5.QtGui import QIcon


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowIcon(QIcon(r'C:\Users\Lhtooo\Desktop\生理信号分析\code\images\bg.jpg'))  # 设置窗口图标
        self.setWindowTitle("眼动工具箱")
        self.setGeometry(300, 150, 1100, 800)

        # 设置窗口大小策略为可调整大小
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 创建控件
        self.button_emotion = QPushButton("情绪分析", self)  # 打开工具箱按钮
        self.button_emotion.clicked.connect(self.open_emotion)
        

        self.button_electronic_scale = QPushButton("电子量表", self)  # 进入电子量表界面按钮
        self.button_electronic_scale.clicked.connect(self.open_electronic_scale)
       

        self.button_strabismus = QPushButton("斜视分析", self)  # 进入斜视界面按钮
        self.button_strabismus.clicked.connect(self.open_strabismus)
        

        # 创建水平布局来放置按钮
        main_layout = QVBoxLayout()

        # 将控件添加到布局中
        main_layout.addWidget(self.button_emotion)
        main_layout.addWidget(self.button_electronic_scale)
        main_layout.addWidget(self.button_strabismus)

        # 添加按钮之间的间隔
        main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
       
        
        # 设置主布局
        self.container = QWidget()
        self.container.setLayout(main_layout)
        self.setCentralWidget(self.container)
    
    def open_emotion(self):
        from emotion import Emotions 
        self.emotion_window = Emotions()
        self.setCentralWidget(self.emotion_window)
        
    
    def open_electronic_scale(self):
        # 这里添加进入电子量表界面的逻辑
        pass
    
    def open_strabismus(self):
        # 这里添加进入斜视界面的逻辑
        from xieshi import Xieshi
        self.xieshi_window = Xieshi()
        self.setCentralWidget(self.xieshi_window)
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
