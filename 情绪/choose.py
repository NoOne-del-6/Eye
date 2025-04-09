import os
import time
import csv
import random
import pygame
from pygame.locals import *
from pupilio import Pupilio
from userlogin import userlogin

def Choose(path, file_name):

    # 初始化Pygame
    pygame.init()
    scn_width, scn_height = (1920, 1080)
    win = pygame.display.set_mode((scn_width, scn_height), FULLSCREEN | HWSURFACE)

    # 实例化眼动追踪器对象

    pupil_io = Pupilio()
    pupil_io.create_session(session_name="deepgaze_demo")
    # pupil_io.calibration_draw(validate=True, hands_free=False, screen=win)
    pupil_io.start_sampling()
    pygame.mouse.set_visible(True)
    pygame.time.wait(100)
    pupil_io.previewer_start('127.0.0.1', 8848)


    # 图片文件夹路径
    img_folder = 'images'
    # images = ['1.jpg', 'white.jpg', '2.jpg', 'white.jpg', '3.jpg', 'white.jpg', '4.jpg']
    images = ['1.jpg', 'white.jpg', '2.jpg']
    # 情绪标签（固定顺序）
    emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

    # 数据保存
    selected_emotions = []  # 用户点击记录
    emotion_layouts = []    # 每张图片的打乱情绪顺序记录

    # 将图片分割成六宫格并打乱
    def split_and_shuffle_image(image):
        img_width, img_height = image.get_width(), image.get_height()
        cell_width, cell_height = img_width // 3, img_height // 2

        # 分割出6个区域
        original_rects = []
        for row in range(2):  # 两行
            for col in range(3):  # 三列
                rect = pygame.Rect(col * cell_width, row * cell_height, cell_width, cell_height)
                original_rects.append(rect)

        # 创建初始绑定（区域、情绪标签、内容）
        bound_data = []
        for rect, emotion in zip(original_rects, emotions):
            cropped_image = image.subsurface(rect).copy()  # 确保裁剪后的图片是独立对象
            bound_data.append((rect, emotion, cropped_image))

        random.shuffle(bound_data)  # 将整体打乱，保持区域、情绪和内容一致
        
        print("打乱后的情绪顺序:", [emotion for _, emotion, _ in bound_data])
        
        return bound_data  # 返回打乱后的数据：[(rect, emotion, image_part), ...]

    # 检测鼠标点击的区域
    def detect_click(pos, display_mapping):
        """
        检测鼠标点击的位置是否落在新的六宫格区域内。
        :param pos: 鼠标点击的屏幕位置 (x, y)
        :param display_mapping: [(新位置 (x, y, width, height), emotion)]
        :return: 匹配的索引
        """
        for idx, ((new_x, new_y, width, height), emotion) in enumerate(display_mapping):
            rect = pygame.Rect(new_x, new_y, width, height)  # 创建新的矩形区域
            if rect.collidepoint(pos):  # 检测点击是否在该矩形内
                return idx  # 返回点击的索引
        return None  # 如果未检测到点击，返回 None


    # 显示六宫格图片
    def display_shuffled_image(bound_data):
        win.fill((0, 0, 0))  # 清空屏幕
        num_cols = 3  # 每行3个格子
        cell_width, cell_height = win.get_width() // num_cols, win.get_height() // 2  # 计算单个格子的宽度和高度

        display_mapping = []  # 保存展示位置和情绪的对应关系

        for idx, (rect, emotion, image_part) in enumerate(bound_data):
            # 重新计算每个图片的绘制位置
            col = idx % num_cols  # 当前列号
            row = idx // num_cols  # 当前行号
            new_x = col * cell_width  # 每列的 x 坐标
            new_y = row * cell_height  # 每行的 y 坐标

            # 在新的位置绘制图片内容
            win.blit(pygame.transform.scale(image_part, (cell_width, cell_height)), (new_x, new_y))  # 缩放到新的格子大小
            pygame.draw.rect(win, (255, 255, 255), (new_x, new_y, cell_width, cell_height), 2)  # 绘制边框

            # 记录新位置和对应情绪
            display_mapping.append(((new_x, new_y, cell_width, cell_height), emotion))


        pygame.display.flip()  # 刷新显示
        return display_mapping  # 返回位置和情绪映射

    
    max_duration = 15

    # 主程序逻辑
    for img_name in images:
        img_path = os.path.join(img_folder, img_name)
        image = pygame.image.load(img_path)
        image = pygame.transform.scale(image, (scn_width, scn_height))  # 缩放图片至全屏

        # 显示white.jpg时跳过交互
        if img_name == 'white.jpg':
            print("--------------------------")
            win.blit(image, (0, 0))
            pygame.display.flip()
            pupil_io.set_trigger(1)
            time.sleep(1)
            continue

        # 切割并打乱六宫格区域与情绪
        bound_data = split_and_shuffle_image(image)
        shuffled_emotions = [emotion for _, emotion, _ in bound_data]
        emotion_layouts.append((img_name, shuffled_emotions))  # 保存情绪顺序
        display_mapping=display_shuffled_image(bound_data)
        
        # 等待用户点击
        start_time=time.time()
        running = True
        selected_idx = None
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pupil_io.stop_sampling()
                    pupil_io.release()
                    pygame.quit()
                    exit()

                if event.type == MOUSEBUTTONDOWN and event.button == 1:  # 左键点击
                    selected_idx = detect_click(event.pos, display_mapping)
                    if selected_idx is not None:
                        selected_emotion = bound_data[selected_idx][1]  # 获取点击的情绪
                        print(f"用户点击了 {img_name} 的情绪：{selected_emotion}")
                        pupil_io.set_trigger(2)
                        selected_emotions.append((img_name, selected_emotion))
                        running = False
            if time.time() - start_time > max_duration:
                    print(f"超时未选择 {img_name}")
                    pupil_io.set_trigger(2)
                    selected_emotions.append((img_name, None))
                    running = False
        # 假如用户没有选择，记录为 None
        if selected_idx is None:
            selected_emotions.append((img_name, None))

    pupil_io.previewer_stop()
    # 停止采样
    pupil_io.stop_sampling()
    # data_dir = "./data"
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir)

    # 保存眼动数据
    eye_data_file = os.path.join(path, file_name)
    pupil_io.save_data(eye_data_file)

    # 保存情绪排列数据
    layout_file = os.path.join(path, "emotion_layouts.csv")
    with open(layout_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Emotion Order"])
        for img, layout in emotion_layouts:
            writer.writerow([img, ', '.join(layout)])

    # 保存用户选择数据
    click_file = os.path.join(path, "selected_emotions.csv")
    with open(click_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Selected Emotion"])
        for img, emotion in selected_emotions:
            writer.writerow([img, emotion])

    print(f"情绪排列数据已保存到 {layout_file}")
    print(f"用户点击数据已保存到 {click_file}")
    print(f"眼动数据已保存到 {eye_data_file}")

    # 释放资源
    pupil_io.release()
    pygame.quit()
