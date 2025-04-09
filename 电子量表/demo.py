'''
    V:基础的量表功能
        逐题展示
    选项打乱: 从左到右 轻度---> 重度 或者 重度 ---> 轻度
'''
import json
import os
import random
import time

import pandas as pd
import pygame
# from openpyxl import load_workbook
from psychopy import core, visual, event

# from pupilio import Pupilio
# from pupilio import DefaultConfig
from userlogin import  userlogin



class Scale:
    def __init__(self, info):
        self.sub_info = info
        self.subj_id = f"{self.sub_info['name']}{self.sub_info['age']}{self.sub_info['gender']}"
        self.title = self.sub_info['type']

        # 初始化pygame
        pygame.init()
        # 关闭鼠标显示
        pygame.mouse.set_visible(True)

        # 屏幕尺寸
        self.SCREEN_WIDTH = 1920
        self.SCREEN_HEIGHT = 1080
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        # self.pi = pi


        # pygame.display.set_caption(self.title)

        # 颜色定义
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)

        # 加载中文字体文件，这里使用系统常见的黑体，你也可以替换为其他喜欢的中文字体文件
        self.font = pygame.font.Font("msyh.ttc", 50)

        # 量表题目
        # os.makedirs()
        _scale_path = os.path.join('.\scale_type', f"{self.title}.xlsx")
        self.questions = self._load_scale_data(_scale_path)
        self.totalScore = 0

        self.data_folder = self.save_title_image()

        with open(os.path.join(self.data_folder, 'sub_info.json'), 'w', encoding='utf-8') as file:
            json.dump(self.sub_info, file, ensure_ascii=False, indent=4)


    # 题目和选项示例数据
    def _load_scale_data(self, filename):
        data_df = pd.read_excel(filename)
        columns = data_df.columns
        # _options_num = (len(columns) - 2)//2

        questions = []
        for i in range(len(data_df)):
            # option shuffle: partially
            _options_shuffle = list(data_df[columns[2::2]].iloc[i])

            # string_len = [len(_options_shuffle[i]) for i in range(len(_options_shuffle))]
            # str_len_max = max(string_len)
            # for j in range(len(_options_shuffle)):
            #     if len(_options_shuffle[j]) < str_len_max:
            #         _temp = _options_shuffle[j]
            #         _options_shuffle[j] = _temp.center(len(_temp) + (str_len_max-len(_temp)))

            _scores_shuffle = list(data_df[columns[3::2]].iloc[i])
            _num_list = [i for i in range(len(_options_shuffle))]

            if random.random() > 0.5:
                _num_list.reverse()
                _options_shuffle = [_options_shuffle[i] for i in _num_list]
                _scores_shuffle = [_scores_shuffle[i] for i in _num_list]

            # _options_shuffle = [f"{i+1}：{_options_shuffle[i]}" for i in range(len(_options_shuffle))]

            question = {
                'num': f"{data_df[columns[0]].iloc[i]}",
                'title': f"问题 {i+1} : {data_df[columns[1]].iloc[i]}",
                'option': _options_shuffle,
                'answer': 0,
                'score': _scores_shuffle
            }
            questions.append(question)
        return questions

    # 绘制背景区域
    def draw_background(self):
        self.screen.fill(self.BLACK)

    # 绘制题目区域
    def draw_title(self, current_question_index, option_rects):
        title_surface = self.font.render((self.questions[current_question_index])['title'], True, self.WHITE)
        title_rect = title_surface.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 5))
        # pygame.draw.rect(screen, BLUE, (title_rect.x - 10, title_rect.y - 10, title_rect.width + 20, title_rect.height + 20))
        # rect = pygame.Rect(0, 0, end_x - 100+10, title_rect.height + 60)
        _rect_coord = self._rect_to_coord(option_rects)
        rect = pygame.Rect(_rect_coord[0], title_rect.y - 30, _rect_coord[2], title_rect.height + 60)

        pygame.draw.rect(self.screen, self.WHITE, rect, width=3)
        self.screen.blit(title_surface, title_rect)
        return rect, title_rect

    # 绘制选项区域
    def draw_options(self, current_question_index):
        option_y = self.SCREEN_HEIGHT // 2
        option_width = (self.SCREEN_WIDTH - 100) // len((self.questions[current_question_index])['option'])

        _option_centers = []
        _width = []
        _height = []

        option_rects = []
        for i, option_text in enumerate((self.questions[current_question_index])['option']):
            option_surface = self.font.render(option_text, True, self.WHITE)
            option_rect = option_surface.get_rect(x=100 + i * option_width, y=option_y)
            self.screen.blit(option_surface, option_rect)

            # _option_center_x, _option_center_y = option_rect.center
            # _option_center = [_option_center_x, _option_center_y]
            _width.append(option_rect.width)
            _height.append(option_rect.height)
            _option_centers.append(option_rect.center)
            option_rects.append(option_rect)

        _width = max(_width)
        _height = max(_height)
        rect_rects = []
        for i, option_center in enumerate(_option_centers):
            # pygame.draw.rect(screen, BLUE, (option_rect.x, option_rect.y, option_rect.width, option_rect.height))
            rect = pygame.Rect(0, 0, _width+20, _height+20)
            rect.center = (option_center[0], option_center[1])

            _answer_option = (self.questions[current_question_index])['answer']

            if _answer_option  and  (i == (_answer_option-1)):
                pygame.draw.rect(self.screen, self.BLUE, rect, width=3)
            else:
                pygame.draw.rect(self.screen, self.WHITE, rect, width=3)

            rect_rects.append(rect)

        return rect_rects, option_rects

    def _rect_to_coord(self, option_rects):
        return [option_rects[0].x, option_rects[0].y, option_rects[-1].x+option_rects[-1].width - option_rects[0].x, option_rects[-1].y+option_rects[-1].height - option_rects[0].y]


    def draw_page(self, text, N):
        self.draw_background()
        # 绘制开始和结束页
        font = pygame.font.Font("msyh.ttc", N)
        end_surface = font.render(text, True, self.WHITE)
        end_rect = end_surface.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        self.screen.blit(end_surface, end_rect)

    def draw_intro(self, lines, N):
        self.draw_background()
        # 绘制开始和结束页
        font = pygame.font.Font("msyh.ttc", N)
        y = self.SCREEN_HEIGHT // 5
        for line in lines:
            end_surface = font.render(line, True, self.WHITE)
            end_rect = end_surface.get_rect(center=(self.SCREEN_WIDTH // 2, y))
            self.screen.blit(end_surface, end_rect)
            y += font.get_linesize()

    def _draw_load_intro(self):
        _intro_path = ".\scale_type"
        _intro_file = [f for f in os.listdir(_intro_path) if f.endswith('.txt') and not f.startswith('~$')]

        _type_intro = f"{self.title}_介绍.txt"
        if _type_intro in _intro_path:
            _file = _type_intro
        else:
            _file = "通用_介绍.txt"
        contents = []
        with open(os.path.join(_intro_path, _file), 'r', encoding='utf-8') as file:
            content = file.readlines()
            for line in content:
                contents.append(line.strip())
        self.draw_intro(contents, 30)
        pygame.display.flip()
        time.sleep(3)

    def main(self):

        self._draw_load_intro()
        self.draw_page('准备开始', 60)
        pygame.display.flip()
        time.sleep(3)

        current_question_index = 0

        # 定义 F 键常量的起始值
        _F_KEY_START = pygame.K_F1

        # self.pi.create_session(session_name='23')

        # # start sampling when the video- is about to play
        # self.pi.start_sampling()  # start sampling
        # core.wait(0.1)  # sleep for 100 ms to capture some extra samples

        # 主循环
        self._running = True
        # self.pi.set_trigger(int(1 + current_question_index))

        while self._running:
            # time.sleep(1)
            _option_F = [_F_KEY_START+i for i in range(len((self.questions[current_question_index])['option']))]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running  = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # 在这里处理按下Esc键的逻辑，例如退出程序
                        self._running = False
                    # if event.key in _option_F:
                    #     for i , _F in enumerate(_option_F):
                    #         if event.key == _F:
                    #             # self.pi.set_trigger(int(1 + current_question_index))
                    #             (self.questions[current_question_index])['answer'] = i+1
                    #
                    #             self.draw_options(current_question_index)
                    #             pygame.display.flip()
                    #             time.sleep(1)
                    #             current_question_index += 1

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()

                    if current_question_index < len(self.questions):
                        # 检查选项点
                        rect_rects, _ = self.draw_options(current_question_index)
                        for i, rect in enumerate(rect_rects):
                            if rect.collidepoint(mouse_pos):
                                # self.pi.set_trigger(int(1 + current_question_index))
                                (self.questions[current_question_index])['answer'] = i+1

                                self.draw_options(current_question_index)
                                pygame.display.flip()
                                time.sleep(1)
                                current_question_index += 1

            if current_question_index < len(self.questions):
                self.draw_background()
                rect_rects, _ = self.draw_options(current_question_index)
                self.draw_title(current_question_index, rect_rects)
            else:
                self._running = False


            pygame.display.flip()

        # self.pi.set_trigger(201)
        # # stop sampling when the video completes
        # core.wait(0.1)  # sleep for 100 ms to capture ending samples
        # self.pi.stop_sampling()  # stop sampling
        # core.wait(0.1)  # sleep for 100 ms to ensure the sampling is closed

        self.draw_page('结束', 60)
        pygame.display.flip()
        time.sleep(2)
        pygame.quit()
        # print(self.questions)
        self.calculation(self.questions)
        # print(self.totalScore)

        # # save the sample data to file
        # file_status = self.save_sample()
        # self.pi.release()

        json_path = os.path.join(self.data_folder, f'{self.title}_question.json')
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(self.questions, file, ensure_ascii=False, indent=4)


    # def save_sample(self):
    #     # save the sample data to file
    #     if not os.path.exists(self.data_folder):
    #         os.makedirs(self.data_folder)
    #     _tmp_file_name = f"{self.title}.csv"
    #     _ret = self.pi.save_data(os.path.join(self.data_folder, _tmp_file_name))
    #
    #     return _ret

    def calculation(self, questions):
        # 量表总分
        for i in range(len(questions)):
            question = questions[i]
            if question['answer']:
                self.totalScore = self.totalScore + (question['score'])[question['answer']-1]
            # self.totalScore = int(self.totalScore * 1.25)

    # def terminate_task(self, title):
    #     """ terminate the task prematurely"""
    #     self.pi.stop_sampling()  # stop sampling
    #     self.pi.release()  # release the tracker object
    #     self.save_sample(title)  # save the sample data to file
    #     core.quit()  # quit psychopy

    def save_title_image(self):
        _data_folder = os.path.join('./data', self.subj_id, self.sub_info['now'])

        _image_path = os.path.join(_data_folder, f'{self.title}_image')
        os.makedirs(_image_path, exist_ok=True)

        page_coordinate = []
        # 保存每道题目的页面
        for i in range(len(self.questions)):
            self.draw_background()
            rect_rects, option_rects = self.draw_options(i)
            rect, title_rect = self.draw_title(i, rect_rects)

            pygame.image.save(self.screen, os.path.join(_image_path, f"{i + 1}.png"))
            page_coordinate.append({"num": f"{i+1}",
                                    "title_rect_coord": [rect.x, rect.y, rect.width, rect.height],
                                    "title_coord": [title_rect.x, title_rect.y, title_rect.width, title_rect.height],
                                    "option_rect_coord":  [[rect_rect.x, rect_rect.y, rect_rect.width, rect_rect.height] for rect_rect in rect_rects],
                                    "option_coord": [[option_rect.x, option_rect.y, option_rect.width, option_rect.height] for option_rect in option_rects]
                                    })


        with open(os.path.join(_image_path, 'info_coord.json'), 'w', encoding='utf-8') as file:
            json.dump(page_coordinate, file, ensure_ascii=False, indent=4)  # indent=4是为了使输出的JSON文件格式化，更易读

        return  _data_folder

if __name__ == '__main__':
    sub_info = userlogin()

    if sub_info["type"] is not None:
        # config = DefaultConfig()
        # config.face_previewing = sub_info ["show_previewer"]  # preview the face image during calibration
        # # initializing the tracker
        #
        # pi = Pupilio(config = config)
        #
        # # calibrate the tracker
        # pi.create_session(session_name="cali")
        # pi.calibration_draw(validate=sub_info["validation"])
        scale = Scale(sub_info)
        scale.main()
    else:
        print("未选择量表")