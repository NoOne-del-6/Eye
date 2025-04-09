import csv
import ctypes
import os.path
import threading
import time
from datetime import datetime

import cv2
import numpy as np

# from pupilio import Pupilio
from pupilio import  ET_ReturnCode
import pupilio
from pupilio import Pupilio

class PreviewThread(threading.Thread):
    def __init__(self, pupil_io, sub_info):
        threading.Thread.__init__(self)
        self._pupil_io = pupil_io
        self._is_running = True
        self.daemon = True
        self.IMG_HEIGHT, self.IMG_WIDTH = 960, 600

        # preview retrieving
        self._preview_img_1 = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH), dtype=np.uint8)
        self._preview_img_2 = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH), dtype=np.uint8)
        self._eye_rects = np.zeros(4 * 4, dtype=np.float32)  # eye bounds
        self._pupil_centers = np.zeros(4 * 2, dtype=np.float32)  # pupil center
        self._glint_centers = np.zeros(4 * 2, dtype=np.float32)  # CR center

        self.preview_compression = [cv2.IMWRITE_JPEG_QUALITY, 40]  # ratio: 0~100
        self.preview_format = ".jpg"
        sub_id = f"{sub_info['name']}{sub_info['age']}{sub_info['gender']}"
        self.save_dir = os.path.join('data', sub_id, sub_info['now'], 'previewer_data')
        os.makedirs(self.save_dir, exist_ok=True)

    def stop(self):
        self._is_running = False
        self.join()
        self._pupil_io = None

    def run(self):
        # 创建CSV文件并写入表头
        csv_filename = os.path.join(self.save_dir, 'frame_times.csv')
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Frame Number', 'System Time (ns)', 'real time'])

        frame_number = 0
        while self._is_running:
            time.sleep(0.016)
            _current_datetime = time.time()
            _current_datetime = int(_current_datetime*1000)
            img_1_ptr = self._preview_img_1.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            img_2_ptr = self._preview_img_2.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

            result = self._pupil_io._et_native_lib.pupil_io_get_previewer(ctypes.pointer(img_1_ptr),
                                                                           ctypes.pointer(img_2_ptr),
                                                                           self._eye_rects, self._pupil_centers,
                                                                           self._glint_centers)

            ctypes.memmove(self._preview_img_1.ctypes.data, img_1_ptr, self._preview_img_1.nbytes)
            ctypes.memmove(self._preview_img_2.ctypes.data, img_2_ptr, self._preview_img_2.nbytes)

            if result == ET_ReturnCode.ET_SUCCESS:
                cv2.imwrite(os.path.join(self.save_dir, f"left_camera_previewer_{frame_number}_{_current_datetime}{self.preview_format}"),
                            self._preview_img_1)
                # cv2.imwrite(os.path.join(self.save_dir, f"right_camera_previewer_{_current_datetime}{self.preview_format}"),
                #             self._preview_img_2)
                # 将帧编号和系统时间写入CSV文件
                with open(csv_filename, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([frame_number, _current_datetime])
                frame_number += 1
        print("Thread shutdown")

    # def run(self):
    #     count = 0
    #     while self._is_running:
    #         time.sleep(0.016)
    #         # # ToDo save video frame
    #         # self.pi.set_trigger(_frame)
    #         # preview_images = self.pi.get_preview_images()
    #         # cv2.imwrite(os.path.join(_img_data_folder, f"{subj_id}_{video_file}_left_camera_img_{_frame}{self.preview_format}"),
    #         #             preview_images[0])
    #         # cv2.imwrite(os.path.join(_img_data_folder, f"{subj_id}_{video_file}_right_camera_img_{_frame}{self.preview_format}"),
    #         #             preview_images[1])
    #         # core.wait(0.01)
    #         _current_datetime = datetime.now()
    #         img_1_ptr = self._preview_img_1.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    #         img_2_ptr = self._preview_img_2.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    #
    #         result = self.pi._et_native_lib.pupil_io_get_previewer(ctypes.pointer(img_1_ptr),
    #                                                                ctypes.pointer(img_2_ptr),
    #                                                                self._eye_rects, self._pupil_centers,
    #                                                                self._glint_centers)
    #
    #         ctypes.memmove(self._preview_img_1.ctypes.data, img_1_ptr, self._preview_img_1.nbytes)
    #         ctypes.memmove(self._preview_img_2.ctypes.data, img_2_ptr, self._preview_img_2.nbytes)
    #
    #         if result == ET_ReturnCode.ET_SUCCESS:
    #             # self.pi.set_trigger(count)
    #             cv2.imwrite(
    #                 os.path.join(self.save_dir, f"left_camera_img_{_current_datetime}{self.preview_format}"),
    #                 self._preview_img_1)
    #             cv2.imwrite(
    #                 os.path.join(self.save_dir, f"right_camera_img_{_current_datetime}{self.preview_format}"),
    #                 self._preview_img_2)
    #             count += 1
    #     print("Thread shutdown")

if __name__ == '__main__':

    pi = pupilio.Pupilio()
    save_dir = 'data/mcf32male/20250325'
    preview_thread = PreviewThread(pupil_io=pi, save_dir=save_dir)
    preview_thread.start()
    pi.create_session("cali")
    pi.start_sampling()
    time.sleep(2)
    pi.stop_sampling()
    preview_thread.stop()
    pi.save_data("demo.csv")
    pi.release()
