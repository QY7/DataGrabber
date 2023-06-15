# -*- coding: utf-8 -*-
from src.DataGrabber.launcher import datagrabber
import traceback
import sys

def excepthook(exc_type, exc_value, traceback_obj):
    # 将异常信息输出到控制台
    traceback.print_exception(exc_type, exc_value, traceback_obj)

    # 保存异常信息到文件
    with open("error.log", "a") as f:
        f.write("=" * 80 + "\n")
        traceback.print_exception(exc_type, exc_value, traceback_obj, file=f)

sys.excepthook = excepthook
datagrabber()