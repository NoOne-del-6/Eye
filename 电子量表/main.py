'''
    量表：眼动
    V2:增加 用户登录时加入演示模式，面部照片抓取，csv文件绝对时间戳转换,
'''
import json
import os
import random
import time
import numpy as np

import pandas as pd
import pygame
from pygame.locals import FULLSCREEN, HWSURFACE
# from openpyxl import load_workbook
from psychopy import core, visual, event

from pupilio import Pupilio
from pupilio import DefaultConfig

from csv_convert import CSVConvert
from thread import PreviewThread
from userlogin import userlogin

# 常量定义
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
FONT_SIZE = 50
FONT_FILE = "msyh.ttc"
FPS = 60


class Scale:
    def __init__(self, pi, info):
        self.intro = None
        self.result_intro = None

        self.sub_info = info
        self.subj_id = f"{self.sub_info['name']}{self.sub_info['age']}{self.sub_info['gender']}"
        self.title = self.sub_info['type']

        # 初始化pygame
        pygame.init()
        # 鼠标显示
        if self.sub_info['model']:
            pygame.mouse.set_visible(True)

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), FULLSCREEN | HWSURFACE)

        self.pi = pi

        # Define the clock for frame rate control
        self.clock = pygame.time.Clock()
        # 字体
        self.font = pygame.font.Font(FONT_FILE, FONT_SIZE)

        # 加载量表题目
        _scale_path = os.path.join('.\scale_type', f"{self.title}.xlsx")
        try:
            self.questions = self._load_scale_data(_scale_path)
        except FileNotFoundError:
            print('错误：未找到量表文件，请检查文件路径。')
        except Exception as e:
            print(f'错误：加载量表文件时发生未知错误：{e}')
        self.totalScore = 0
        self.data_folder = self.save_title_image()

    def _load_scale_data(self, filename):
        """加载量表数据"""
        data_df = pd.read_excel(filename, sheet_name='题目')
        columns = data_df.columns
        # _options_num = (len(columns) - 2)//2

        questions = []
        for i in range(len(data_df)):
            # option shuffle: partially
            _options_shuffle = list(data_df[columns[2::2]].iloc[i])

            _scores_shuffle = list(data_df[columns[3::2]].iloc[i])
            _num_list = [i for i in range(len(_options_shuffle))]

            if random.random() > 0.5:
                _num_list.reverse()
                _options_shuffle = [_options_shuffle[i] for i in _num_list]
                _scores_shuffle = [_scores_shuffle[i] for i in _num_list]
            if self.sub_info['model'] == 0:
                _options_shuffle = [f"{i + 1}：{_options_shuffle[i]}" for i in range(len(_options_shuffle))]

            question = {
                'num': f"{data_df[columns[0]].iloc[i]}",
                'title': f"问题 {i + 1} : {data_df[columns[1]].iloc[i]}",
                'option': _options_shuffle,
                'answer': 0,
                'score': _scores_shuffle
            }
            questions.append(question)
        info_df = pd.read_excel(filename, sheet_name='简介')

        self.intro = info_df['量表简介'].iloc[0]
        self.result_intro = info_df['结果判断'].iloc[0]

        return questions

    def _draw_background(self):
        """绘制背景区域"""
        self.screen.fill(BLACK)

    def _draw_title(self, current_question_index):
        """绘制题目区域"""
        title = self.questions[current_question_index]['title']
        title_surface = self.font.render(title, True, WHITE)
        title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 5))
        # pygame.draw.rect(screen, BLUE, (title_rect.x - 10, title_rect.y - 10, title_rect.width + 20, title_rect.height + 20))
        rect = pygame.Rect(0, 0, SCREEN_WIDTH - 200, title_rect.height + 60)
        rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 5)

        pygame.draw.rect(self.screen, WHITE, rect, width=3)
        self.screen.blit(title_surface, title_rect)
        return rect, title_rect

    # 绘制选项区域
    def _draw_options(self, current_question_index):
        options = (self.questions[current_question_index])['option']
        option_num = len(options)

        option_y = SCREEN_HEIGHT // 2
        option_width = (SCREEN_WIDTH - 200) // option_num

        option_centers = []
        _width = []
        _height = []
        x = [i + 100 + (option_width // 2) for i in range(0, (SCREEN_WIDTH - 200), option_width)]
        option_rects = []
        for i, option_text in enumerate((self.questions[current_question_index])['option']):
            option_surface = self.font.render(option_text, True, WHITE)
            option_rect = option_surface.get_rect(center=(x[i], option_y))

            self.screen.blit(option_surface, option_rect)

            _width.append(option_rect.width)
            _height.append(option_rect.height)
            option_centers.append(option_rect.center)
            option_rects.append(option_rect)

        _width = max(_width)
        _height = max(_height)
        rect_rects = []
        for i, option_center in enumerate(option_centers):
            rect = pygame.Rect(0, 0, _width + 20, _height + 20)
            rect.center = (option_center[0], option_center[1])

            _answer_option = (self.questions[current_question_index])['answer']

            if _answer_option and (i == (_answer_option - 1)):
                pygame.draw.rect(self.screen, BLUE, rect, width=3)
            else:
                pygame.draw.rect(self.screen, WHITE, rect, width=3)

            rect_rects.append(rect)

        return rect_rects, option_rects

    def _draw_text(self, text, N):
        """绘制文本"""
        self._draw_background()
        font = pygame.font.Font(FONT_FILE, N)

        if isinstance(text, list):
            y = SCREEN_HEIGHT // 2 - (len(text) // 2) * font.get_linesize()
            for line in text:
                end_surface = font.render(line, True, WHITE)
                end_rect = end_surface.get_rect(center=(SCREEN_WIDTH // 2, y))
                self.screen.blit(end_surface, end_rect)
                y += font.get_linesize()
        elif isinstance(text, str):
            end_surface = font.render(text, True, WHITE)
            end_rect = end_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(end_surface, end_rect)

    def show_long_text(self, index):
        """显示文本信息"""
        if index == "intro":
            content = str(self.intro)
        elif index == 'result':
            content = str(self.result_intro)

        contents = []

        if content != 'nan':
            content = content.split('\n')
            for line in content:
                contents.append(line.strip())

        if index == 'intro':
            _intro = "提示：选项顺序会随机打乱，请仔细阅读选项内容，用鼠标或键盘完成作答"
            contents.append(_intro.strip())
        elif index == 'result':
            contents.insert(0, f"本次量表测试评分：{self.totalScore}")

        self._draw_text(contents, 30)
        pygame.display.flip()
        pygame.time.wait(3000)

    def main(self):
        """主循环"""
        self.show_long_text('intro')
        # self._draw_text('准备开始', 60)# 开始页面
        # pygame.display.flip()

        current_question_index = 0
        _F_KEY_START = pygame.K_1  # 定义 F 键常量的起始值

        self.pi.start_sampling()  # start sampling
        pygame.time.wait(100)  # sleep for 100 ms to capture some extra samples
        # 主循环
        running = True
        self.pi.set_trigger(int(1 + current_question_index))

        while running:
            _option_F = [_F_KEY_START + i for i in range(len((self.questions[current_question_index])['option']))]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # 在这里处理按下Esc键的逻辑，例如退出程序
                        running = False
                    elif event.key in _option_F:
                        for i, _F in enumerate(_option_F):
                            if event.key == _F:
                                self.handle_answer(current_question_index, i + 1)
                                current_question_index += 1

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if current_question_index < len(self.questions):
                        rect_rects, _ = self._draw_options(current_question_index)
                        for i, rect in enumerate(rect_rects):
                            if rect.collidepoint(mouse_pos):
                                self.handle_answer(current_question_index, i + 1)
                                current_question_index += 1

            if current_question_index < len(self.questions):
                self._draw_background()
                self._draw_options(current_question_index)
                self._draw_title(current_question_index)
            else:
                running = False

            if self.sub_info['show_gaze']:
                # get the newest gaze position
                left, right, bino = self.pi.get_current_gaze()
                status, gx, gy = bino
                gx = int(gx)
                gy = int(gy)
                pygame.draw.circle(self.screen, (0, 255, 0), (gx, gy), 50, 5)  # cursor for the bino gaze

            self.clock.tick(60)  # refresh the screen at 60 fps
            pygame.display.flip()

        self.pi.set_trigger(201)
        # stop sampling when the video completes
        pygame.time.wait(100)  # sleep for 100 ms to capture ending samples
        self.pi.stop_sampling()  # stop sampling
        pygame.time.wait(100)  # sleep for 100 ms to ensure the sampling is closed

        # score and show
        # self._draw_text('结束', 60)
        self.calculation(self.questions)
        self.show_long_text('result')

        pygame.quit()

        # save the sample data to file
        file_status, _tmp_file_path = self.save_sample()
        # self.pi.release()

        # 保存题目信息
        question_json_path = os.path.join(self.data_folder, f'{self.title}_image', f'{self.title}_question.json')
        with open(question_json_path, 'w', encoding='utf-8') as file:
            json.dump(self.questions, file, ensure_ascii=False, indent=4)

        # 保存用户信息
        sub_info_json_path = os.path.join(self.data_folder, 'sub_info.json')
        with open(sub_info_json_path, 'w', encoding='utf-8') as file:
            json.dump(self.sub_info, file, ensure_ascii=False, indent=4)
        return _tmp_file_path

    def handle_answer(self, current_question_index, answer_index):
        """处理用户的答案"""
        self.pi.set_trigger(int(1 + current_question_index))
        self.questions[current_question_index]['answer'] = answer_index
        self._draw_options(current_question_index)
        pygame.display.flip()
        pygame.time.wait(1000)

    def save_sample(self):
        """保存眼动数据"""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        _tmp_file_name = f"{self.subj_id}_{self.sub_info['now']}_{self.title}.csv"
        _tmp_file_path = os.path.join(self.data_folder, _tmp_file_name)
        _ret = self.pi.save_data(_tmp_file_path)
        return _ret, _tmp_file_path

    def calculation(self, questions):
        """计算总得分"""
        for i in range(len(questions)):
            question = questions[i]
            if question['answer']:
                self.totalScore = self.totalScore + (question['score'])[question['answer'] - 1]
        self.sub_info['score'] = self.totalScore

    # def terminate_task(self, title):
    #     """ terminate the task prematurely"""
    #     self.pi.stop_sampling()  # stop sampling
    #     self.pi.release()  # release the tracker object
    #     self.save_sample(title)  # save the sample data to file
    #     core.quit()  # quit psychopy

    def save_title_image(self):
        """保存页面截图和坐标信息"""
        _data_folder = os.path.join('./data', self.subj_id, self.sub_info['now'])

        _image_path = os.path.join(_data_folder, f'{self.title}_image')
        os.makedirs(_image_path, exist_ok=True)

        page_coordinate = []
        # save page
        for i in range(len(self.questions)):
            self._draw_background()
            rect_rects, option_rects = self._draw_options(i)
            rect, title_rect = self._draw_title(i)

            pygame.image.save(self.screen, os.path.join(_image_path, f"{i + 1}.png"))
            page_coordinate.append({"num": f"{i + 1}",
                                    "title_rect_coord": [rect.x, rect.y, rect.width, rect.height],
                                    "title_coord": [title_rect.x, title_rect.y, title_rect.width, title_rect.height],
                                    "option_rect_coord": [[rect_rect.x, rect_rect.y, rect_rect.width, rect_rect.height]
                                                          for rect_rect in rect_rects],
                                    "option_coord": [
                                        [option_rect.x, option_rect.y, option_rect.width, option_rect.height] for
                                        option_rect in option_rects]
                                    })

        # 保存坐标信息
        coord_json_path = os.path.join(_image_path, 'info_coord.json')
        with open(coord_json_path, 'w', encoding='utf-8') as f:
            json.dump(page_coordinate, f, ensure_ascii=False, indent=4)

        return _data_folder


if __name__ == '__main__':

    sub_info, inputDlg = userlogin()

    if inputDlg.OK:
        if sub_info["type"] is not None:
            # config
            config = DefaultConfig()
            config.face_previewing = sub_info["show_previewer"]  # preview the face image during calibration

            # initializing the tracker
            pi = Pupilio(config=config)

            # calibrate the tracker
            pi.create_session(session_name="cali")
            pi.calibration_draw(validate=sub_info["validation"])

            # previewer
            preview_thread = PreviewThread(pupil_io=pi, sub_info=sub_info)
            preview_thread.start()

            scale = Scale(pi, sub_info)
            csv_name = scale.main()
            
            preview_thread.stop()
            pi.release()

            # convert_timestamp_to_system_time
            csvconvert = CSVConvert()
            csvconvert.convert(csv_name)


        else:
            print("未选择量表")