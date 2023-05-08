# -*- coding: utf-8 -*-
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from .datagrabber import mywindow
import keyboard
from .datagrabber import ICON_LOGO


def datagrabber():
    app = QApplication(sys.argv)
    #MainWindow = QMainWindow()
    window = mywindow()
    window.show()
    keyboard.on_press(window.keyboardEventReceived)
    window.setWindowTitle("DataGrabber")
    window.setWindowIcon(QIcon(ICON_LOGO))
    sys.exit(app.exec_())

if __name__ == '__main__':
    
    try:
        datagrabber()
    except Exception as e:
        print(e)