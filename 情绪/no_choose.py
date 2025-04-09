import os
import time
import pygame
from pygame.locals import *
from pupilio import Pupilio, DefaultConfig
import psutil
import pandas as pd
import datetime


def Nochoose(path, file_name):
    
   
    # use the Pygame library for graphics, first init pygame and open a full screen window
    pygame.init()
    scn_width, scn_height = (1920, 1080)
    win = pygame.display.set_mode((scn_width, scn_height), FULLSCREEN|HWSURFACE)

    # use a custom config file to control the tracker
    config = DefaultConfig()

    # Heuristic filter, default look_ahead = 2 (i.e., a noisy spike is determined by
    # 4 flanking samples)
    config.look_ahead = 2

    # instantiate a tracker object
    pupil_io = Pupilio(config)

    # create a task session, and set a session name
    # The session name must contain only letters, digits or underscores without any special characters.
    pupil_io.create_session(session_name="deepgaze_demo")

    # 校验
    # pupil_io.calibration_draw(validate=True, hands_free=False, screen=win)

    #  start retrieving gaze
    pupil_io.start_sampling()
    pygame.time.wait(100)  # sleep for 100 ms so the tracker cache some sample

    pupil_io.previewer_start('127.0.0.1', 8848)
    pupil_io.set_trigger(1)

    img_folder = 'images'
    
    
    images = ['1.jpg', '2.jpg']

    # images = ['1.jpg', '2.jpg', '3.jpg','4.jpg','5.jpg','6.jpg']
    # images = ['1.jpg', '2.jpg', '3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg']
    # images = ['1.jpg', '2.jpg', '3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg','11.jpg','12.jpg','13.jpg','14.jpg','15.jpg','16.jpg']
   



    # show the images one by one in a loop, press a ENTER key to exit the program
    for _img in images:
        # show the image on screen
        win.fill((128, 128, 128))
        im = pygame.image.load(os.path.join(img_folder, _img))
        win.blit(im, (0, 0))
        pygame.display.flip()
        # send a trigger to record in the eye movement data to mark picture onset
        pupil_io.set_trigger(202)

        # now lets show the gaze point, press any key to close the window
        got_key = False
        max_duration = 5000
        t_start = pygame.time.get_ticks()
        pygame.event.clear()  # clear all cached events if there were any
        gx, gy = -65536, -65536
    
        
        
        
        
        
        while not (got_key or (pygame.time.get_ticks() - t_start)>=max_duration):   # 看是否有按键或者时间超出15秒
            # get the newest gaze position
            left, right, bino = pupil_io.get_current_gaze()   # 得到最新的眼球视线坐标
            status, gx, gy = bino
            gx = int(gx)
            gy = int(gy)

            # check key press 检查是否有按键 
            for ev in pygame.event.get():
                if ev.type == KEYDOWN:
                    if ev.key == K_RETURN:
                        got_key = True
                
            win.blit(im, (0,0))
            pygame.draw.circle(win, (0, 255, 0), (gx, gy), 50, 5)  # cursor for the left eye
            pygame.display.flip()

        
        
        
    # stop sampling
    pygame.time.wait(100)  # sleep for 100 ms to capture ending samples
    pupil_io.set_trigger(2)
    pupil_io.previewer_stop()
    pupil_io.stop_sampling()

    
        
        
        
    
    try:
       pupil_io.save_data(os.path.join(path, file_name))
    except Exception as e:
       print(f"保存数据时发生错误: {e}")
       

    # release the tracker instance
    pupil_io.release()

    # quit pygame
    pygame.quit()
    
    # 将原始文件的时间戳转换为系统时间
    # 获取电脑启动时间（时间戳，秒）
    boot_time_timestamp = int(psutil.boot_time())

    # 获取启动时间
    boot_time = datetime.datetime.fromtimestamp(boot_time_timestamp)

    # 读取CSV文件
    csv_filename = os.path.join(path, file_name) # 替换为你的文件名
    df = pd.read_csv(csv_filename)

    # 假设时间戳列名为 "timestamp_ns"，你可以根据实际的列名修改
    def convert_timestamp_to_system_time(timestamp_ns):
        # 将纳秒时间戳转换为秒
        timestamp_seconds = int(timestamp_ns / 1_000_000_000)

        # 计算运行时间（即时间戳对应的秒数）
        uptime = datetime.timedelta(seconds=timestamp_seconds)

        # 计算系统时间（启动时间 + 运行时间）
        system_time = boot_time + uptime

        # 获取纳秒部分
        nanoseconds = timestamp_ns % 1_000_000_000

        formatted_time_with_ns = f"{system_time}.{nanoseconds:09d}" # 添加纳秒部分

        return formatted_time_with_ns


    # 使用apply方法将新列添加到DataFrame中
    df['real_system_time'] = df['timestamp'].apply(convert_timestamp_to_system_time)

    # 将新的数据保存到原CSV文件中
    df.to_csv(csv_filename, index=False)

    print("新列已成功添加，并保存为 '{csv_filename}'")
