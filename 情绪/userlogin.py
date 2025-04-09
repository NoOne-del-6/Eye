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
    sub_info = {'姓名': '', 
                '性别': ['男', '女'],
                '年龄': '',
                '情绪范式': ["情绪图片", "人脸图片"]}
    inputDlg = gui.DlgFromDict(dictionary=sub_info, title='用户信息',
                               order=['姓名', '性别', '年龄', '情绪范式'])

    _current_datetime = datetime.now()
    _current_time = _current_datetime.strftime("%Y%m%d")
    expInfo = {
        "name": "" if sub_info['姓名'] is None else sub_info['姓名'],
        "age": "18" if sub_info['年龄'] is None else sub_info['年龄'],
        "gender": "male" if sub_info['性别'] == '男' else "female",
        'mode': "情绪图片" if sub_info['情绪范式'] == '情绪图片' else "人脸图片",
        "now": _current_time
        
    }

    return  expInfo, inputDlg
if __name__ == '__main__':
    sub_info, inputDlg= userlogin()
    if inputDlg.OK:
        print(sub_info)
    else:
        print('cancel')
