#!/usr/bin/python3
# -*- coding: utf-8 -*-
from datetime import datetime
import os
import pygame

# from pupilio import Pupilio
# from pupilio import DefaultConfig
from psychopy import gui, visual, core, event


def userlogin():
    # 收集被试的基本信息
    sacle_path = '.\scale_type'
    scale_type = [f.strip(".xlsx") for f in os.listdir(sacle_path) if f.endswith('.xlsx') and not f.startswith('~$')]
    sub_info = {'姓名': '', '性别': ['男', '女'],
                '年龄': '', '量表类型': scale_type,
                '演示模式':['否', '是'],
                '面容预览':['否', '是'],
                '眼动验证':['否', '是'],
                '完成模式':['鼠标', '键盘']}
    inputDlg = gui.DlgFromDict(dictionary=sub_info, title='用户信息',
                               order=['姓名', '性别', '年龄', '量表类型', '面容预览', '眼动验证', '完成模式', '演示模式'])

    _current_datetime = datetime.now()
    _current_time = _current_datetime.strftime("%Y%m%d")
    expInfo = {
        "name": "" if sub_info['姓名'] is None else sub_info['姓名'],
        "age": "18" if sub_info['年龄'] is None else sub_info['年龄'],
        "gender": "male" if sub_info['性别'] == '男' else "female",
        "show_previewer": 1 if sub_info['面容预览'] == '是' else 0,
        "show_gaze": 1 if sub_info['演示模式'] == '是' else 0,
        "validation": 1 if sub_info['眼动验证'] == '是' else 0,
        "type": sub_info['量表类型'],
        "model": 1 if sub_info['完成模式'] == '鼠标' else 0,
        "score": None,
        "now": _current_time
    }

    return  expInfo, inputDlg
if __name__ == '__main__':
    sub_info, inputDlg= userlogin()
    if inputDlg.OK:
        print(sub_info)
    else:
        print('cancel')
