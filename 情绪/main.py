import datetime
import os
import threading
from time import sleep
import cv2
from no_choose import Nochoose
from choose import Choose
from get_video import get_video
from userlogin import userlogin  

# 创建文件夹（如果不存在的话）
def create_folder(save_folder):  
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


def main():
    # path='data/video'

    sub_info, inputDlg = userlogin()
    path = 'data'
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # data_dir =  os.path.normpath(os.path.join('C:/Users/DeepGaze/Desktop/working/情绪范式/data', sub_info['name'],sub_info['now']))
    data_dir =  os.path.normpath(os.path.join(path, sub_info['mode'],sub_info['name']+'_'+ sub_info['gender'],current_time))
    print(f"目标目录路径: {data_dir}")
    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)  # 创建目录
            print(f"目录已创建: {data_dir}")
        else:
            print(f"目录已存在: {data_dir}")
    except PermissionError:
        print(f"没有权限在此目录 {data_dir} 下创建文件夹。")
        exit(1)  # 如果没有权限，退出程序
    except Exception as e:
        print(f"创建目录时发生错误: {e}")
        exit(1)     
        
    file_name = f"{sub_info['name']}.csv"  # Use user's name in the CSV file name    
        
   

    if (sub_info['mode'] == "人脸图片"):
        # 启动眼动线程
        choose_thread = threading.Thread(target=Choose, args=(data_dir, file_name))
        choose_thread.daemon = True  
        choose_thread.start()
        print('启动眼动线程')
    
    if (sub_info['mode'] == "情绪图片"):
        # 启动眼动线程
        nochoose_thread = threading.Thread(target=Nochoose, args=(data_dir, file_name))
        nochoose_thread.daemon = True  
        nochoose_thread.start()
        print('启动眼动线程')
    
    # 启动视频捕捉的线程
    get_video_thread = threading.Thread(target=get_video, args=(data_dir,))
    get_video_thread.daemon = True  
    get_video_thread.start()
    print('启动视频捕捉的线程')

    # 等待前两个线程结束
    if (sub_info['mode'] == "情绪图片"):
        choose_thread.join()
    if (sub_info['mode'] == "人脸图片"):
        nochoose_thread.join()
    get_video_thread.join()

   

if __name__ == "__main__":
    main()