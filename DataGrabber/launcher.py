# -*- coding: utf-8 -*-
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from .datagrabber import mywindow
import keyboard
from .datagrabber import ICON_LOGO
import traceback

def excepthook(exc_type, exc_value, traceback_obj):
    # 将异常信息输出到控制台
    traceback.print_exception(exc_type, exc_value, traceback_obj)

    # 保存异常信息到文件
    with open("error.log", "a") as f:
        f.write("=" * 80 + "\n")
        traceback.print_exception(exc_type, exc_value, traceback_obj, file=f)

def datagrabber():
    app = QApplication(sys.argv)
    #MainWindow = QMainWindow()
    window = mywindow()
    window.show()
    keyboard.on_press(window.keyboardEventReceived)
    window.setWindowTitle("DataGrabber")
    window.setWindowIcon(QIcon(ICON_LOGO))
    sys.excepthook = excepthook
    sys.exit(app.exec_())

if __name__ == '__main__':
    try:
        datagrabber()
    except Exception as e:
        print(e)