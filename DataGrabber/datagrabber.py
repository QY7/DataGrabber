# -*- coding: utf-8 -*-
from asyncio.windows_events import NULL
from PIL import Image
from PIL import ImageGrab
from numpy import array,savetxt, size,tile,newaxis
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .ui import Ui_MainWindow
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from enum import Enum
import os
from .find_contour import find_box
import json
from DataGrabber import _root_path

def get_path(filename):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    else:
        return os.path.dirname(__file__)+f'/img/{filename}'

ICON_PICKER = get_path('picker.png')
ICON_ERASER = get_path('eraser_rect.png')
ICON_LOGO = get_path('logo.ico')
class AxisType(Enum):
    LINEAR = 1
    LOG = 2

class SystemState(Enum):
    IDLE = 0
    ERASING = 1
    PICKING_COLOR = 2
    POS_LEFT = 3
    POS_RIGHT = 4

class mywindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.init_params()
        self.auto_box = False
        self.pushButton_preview.clicked.connect(self.plot_value)
        self.pushButton_picker.clicked.connect(self.color_picker)
        self.pushButton_eraser.clicked.connect(self.eraser)
        self.pushButton_load.clicked.connect(self.load_img_from_clipboard)
        self.pushButton_add.clicked.connect(self.add_curve)
        self.horizontalSlider_eraser.valueChanged.connect(self.change_eraser)
        self.actionTXT.triggered.connect(self.export_data)
        self.actionCSV.triggered.connect(self.export_data_csv)
        # self.menuExport_to_Excel.triggered.connect(self.export_data_excel)
        self.actionImport.triggered.connect(self.import_img)
        # self.setaxis.clicked.connect(self.tailor_img)
        self.leftbottom.clicked.connect(self.set_to_left)
        self.righttop.clicked.connect(self.set_to_right)
        self.label_img.mousePressEvent = self.get_pos
        self.auto_axis.clicked.connect(self.auto_mode)
        self.label_img.mouseMoveEvent = self.erasing

        self.pushButton_eraser.setEnabled(False)
        self.pushButton_picker.setEnabled(False)
        self.pushButton_preview.setEnabled(True)
        self.leftbottom.setEnabled(False)
        self.righttop.setEnabled(False)
        self.auto_axis.setEnabled(False)
        # self.setaxis.setEnabled(False)
        # self.Eraser_size.valueChanged.connect(self.eraser_size_change)
        # self.Grid_size.valueChanged.connect(self.grid_size_change)
        
        self.pos_left_bottom = [0,0]
        self.pos_right_top = [0,0]
        self.setting = QSettings(f'{os.path.dirname(__file__)}/.temp', QSettings.IniFormat) 
        self.spacing = 1
        self.has_frame = False
        self.has_mask = None
        print(f"{_root_path}/config.json")
        with open(f"{_root_path}/config.json",'r') as f:
            self.system_config = json.load(f)
        print(self.system_config)
        
    def load_img_from_file(self,file):
        try:
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
        except:
            print("Error Loading")
        
        img = self.image_resize(img)
        self.image_height,self.image_width = img.shape[0:2]
        self.current_img = img

        self.reset_state()
        self.refresh_img()

    def load_img_from_clipboard(self):
        try:
            img_raw = ImageGrab.grabclipboard()
            img_cvted = cv2.cvtColor(np.array(img_raw),cv2.COLOR_RGB2BGR)
        except TypeError as e:
            print(e)
            return
        except:
            return

        if len(array(img_cvted).shape)!=3:
            return
        # 重新变大小
        img_resized = self.image_resize(img_cvted)
        self.current_img = img_resized
        self.image_height,self.image_width = img_resized.shape[0:2]

        self.reset_state()
        self.refresh_img()

    def reset_state(self):
        self.label_info.setText("Fill in axis and set axis")
        self.leftbottom.setEnabled(True)
        self.righttop.setEnabled(True)
        self.auto_axis.setEnabled(True)
        self.pushButton_picker.setEnabled(True)
        self.pushButton_eraser.setEnabled(True)

        # restore the mousepress on image to position get
        self.label_img.mousePressEvent = self.get_pos

        self.result_list = {}
        self.curve_idx = 0
        self.has_frame = False
        self.has_mask = False

        self.system_state = SystemState.IDLE


    def set_spacing(self):
        self.spacing = self.horizontalSlider_spacing.value()
        print(f"Spacing has been set to {self.spacing}")

    def image_resize(self,image):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]
        # Maximum height should be 500 and maximum width should be 1000
        if(w / h > 2):
            _width = self.system_config["max_width"]
            _height = int(_width/w*h)
        else:
            _height = self.system_config["max_height"]
            _width = int(_height/h*w)

        dim = (_width,_height)
        # resize the image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        # return the resized image
        return resized
        
    def set_color(self,color):
        """将pick的颜色在显示区域显示

        Args:
            color ([type]): [description]
        """
        self.color_preview.setStyleSheet(f'QPushButton {{background-color: {color};}}')

    def update_blender(self):
        """融合self.mask和self.img，并加以显示
        """
        
        self.refresh_img()

    def read_config(self):
        """读取panel设置

        Returns:
            Boolean: True读取成功
        """
        try:
            self.startX = float(self.lineEdit_xmin.text())
            self.endX = float(self.lineEdit_xmax.text())
            self.startY = float(self.lineEdit_ymin.text())
            self.endY = float(self.lineEdit_ymax.text())
            self.xaxis_type = AxisType.LOG if self.checkBox_logx.isChecked() else AxisType.LINEAR
            self.yaxis_type = AxisType.LOG if self.checkBox_logy.isChecked() else AxisType.LINEAR
            
            if(self.xaxis_type == AxisType.LOG and (self.startX == 0 or self.endX == 0)):
                raise AssertionError
            if(self.yaxis_type == AxisType.LOG and (self.startY == 0 or self.endY == 0)):
                raise AssertionError
            
            assert (self.endX>self.startX)
            assert (self.endY >self.startY)
            assert ((self.xaxis_type is AxisType.LINEAR) or (self.xaxis_type is AxisType.LOG and (self.startX>0)))
            assert ((self.yaxis_type is AxisType.LINEAR) or (self.yaxis_type is AxisType.LOG and (self.startY>0)))
            
            return True
        
        except Exception:
            QMessageBox.warning(self,"Warning","Invalid input!")
            return False
    
    def refresh_img(self):
        """刷新img的显示图像

        Args:
            img ([type]): [description]
        """
        height, width, _ = self.current_img.shape
        bytesPerLine = 3 * width
        img_to_show = self.current_img.copy()


        # If color mask existed, add mask to it
        if(self.has_mask):
            if(self.mask.ndim==2):
                # When used to merge with the image, the mask has to be 3-dimmension
                self.show_mask = tile(self.mask[:,:,np.newaxis],3)
            else:
                self.show_mask = self.mask
            # Morph with mask
            img_to_show=cv2.addWeighted(img_to_show,self.system_config['blend_transparency'],self.show_mask,1,0)

        # If frame existed, add frame to it
        if(self.has_frame):
            x1 = self.pos_left_bottom[0]
            y1 = self.pos_right_top[1]
            x2 = self.pos_right_top[0]
            y2 = self.pos_left_bottom[1]
            cv2.rectangle(img_to_show, (x1, y1), (x2, y2), (36,255,12), 1)
        qImg = QtGui.QImage(img_to_show.tobytes(), width, height, bytesPerLine, QtGui.QImage.Format_BGR888)
        self.label_img.setPixmap(QtGui.QPixmap(qImg))
        return

    def set_to_left(self,event):
        self.has_frame = True
        self.label_img.unsetCursor()
        self.system_state = SystemState.POS_LEFT

    def set_to_right(self,event):
        self.has_frame = True
        self.label_img.unsetCursor()
        self.system_state = SystemState.POS_RIGHT

    def get_pos(self,event):
        print(self.system_state)
        if(self.system_state != SystemState.POS_LEFT and self.system_state != SystemState.POS_RIGHT):
            return
        if(self.system_state == SystemState.POS_LEFT):
            self.pos_left_bottom = (event.pos().x(),event.pos().y())
            print("left {0},{1}".format(event.pos().x(),event.pos().y()))
            self.refresh_img()
        elif((self.system_state == SystemState.POS_RIGHT)):
            self.pos_right_top = (event.pos().x(),event.pos().y())
            print("right {0}".format(event.pos()))
        self.refresh_img()

    def set_fuzzy_color_range(self,pixdata):
        """根据pixdata获取一个这个颜色附近的区间

        Args:
            pixdata (np.array BGR): BGR color
        """
        color = cv2.cvtColor(np.uint8([[pixdata]]),cv2.COLOR_BGR2HSV)
        color_pixel = color[0][0]
        # print(color)

        self.lower_color_lim = array([color_pixel[0]-10,color_pixel[1]-50,color_pixel[2]-20])
        self.higher_color_lim = array([color_pixel[0]+10,color_pixel[1]+50,color_pixel[2]+20])

    def get_color_mask(self,img):
        """根据颜色的范围获取mask

        Args:
            img ([type]): [description]

        Returns:
            [type]: [description]
        """
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,self.lower_color_lim,self.higher_color_lim)
        return mask

    def erasing(self,event):
        """获取event的坐标，擦除mask在坐标附近的值

        Args:
            event ([type]): [description]
        """
        if((event.buttons() & QtCore.Qt.LeftButton) and (self.system_state == SystemState.ERASING)):
            # 由mousemove触发，判定事件为左键按下时即可知道发生了拖动event
            # 判定系统状态为erasing，说明此时为擦除状态
            # 首先获取x,y的坐标位置
            x = event.pos().x()
            y = event.pos().y()
            # print(f"Erasing! at [{x},{y}]")
            if(x<0):
                x = 0
            elif (x>=self.image_width-1):
                x = self.image_width-1
            if(y<0):
                y = 0
            elif(y>self.image_height-1):
                y = self.image_height-1

            # 擦除erasing_range范围内的点
            y_range_up = y-self.erase_range
            y_range_down = y+self.erase_range
            x_range_left = x-self.erase_range
            x_range_right = x+self.erase_range

            y_range_up = y_range_up if(y_range_up>=0) else 0
            y_range_down = y_range_down if(y_range_down<=self.image_height) else self.image_height
            x_range_left = x_range_left if(x_range_left>=0) else 0
            x_range_right = x_range_right if(x_range_right<=self.image_width) else self.image_width
            # 更新mask
            self.mask[y_range_up:y_range_down,x_range_left:x_range_right]=0
            # 更新显示的图
            self.update_blender()

    def extract_data(self,data):
        result = []
        # Here the width and height
        curve_range_height,curve_range_width = data.shape

        for i in range(0,curve_range_width,1):
            # 如果有黑色像素，取中值
            # 注意行列和矩阵的维度的对应
            # 矩阵中，第一维为行，第二为列。所以对x坐标进行循环， 就应该取出对应的列的所有行
            if(255 in data[:,i]):
                ldx = curve_range_height-np.median(np.argwhere(data[:,i]==255))
                result.append([i,ldx])
        # print(array(result))
        return array(result)

    def pick_color(self,event):
        if(self.system_state == SystemState.PICKING_COLOR):
            x = event.pos().x()
            y = event.pos().y()
            self.color_set = self.current_img[y][x]
            self.pushButton_preview.setEnabled(True)
            self.pushButton_eraser.setEnabled(True)
            color = '#'
            color += str(hex(self.color_set[2]))[-2:].replace('x', '0').upper()
            color += str(hex(self.color_set[1]))[-2:].replace('x', '0').upper()
            color += str(hex(self.color_set[0]))[-2:].replace('x', '0').upper()
            self.set_color(color)
            self.color_set_hex = color
            print("当前选择颜色为:")
            print(self.color_set)
            # 设置当前的颜色选择范围
            self.set_fuzzy_color_range(self.color_set)
            self.mask = self.get_color_mask(self.current_img)
            self.has_mask = True
            self.update_blender()
            self.pushButton_preview.setEnabled(True)

    def init_params(self):
        self.label_img.unsetCursor()
        self.result = None

        self.lower_color_lim = [0,0,0]
        self.higher_color_lim = [255,255,255]

        self.image_width = 0
        self.image_height = 0
        self.startX = 0 # x坐标最小
        self.startY = 1 # x坐标最大
        self.endX =0 # y坐标最小
        self.endY = 1 # y坐标最大

        self.xaxis_type = AxisType.LINEAR
        self.yaxis_type = AxisType.LINEAR

        self.mask = None
        self.color_set = None
        self.erase_range = self.horizontalSlider_eraser.value()

        self.label_info.setText("Load image")

    def data_mapping(self,data,width,height):
        data_spaced_x = data[::self.spacing,0]
        data_spaced_y = data[::self.spacing,1]
        if(self.xaxis_type == AxisType.LINEAR):
            data_spaced_x = data_spaced_x/width*(self.endX-self.startX)+self.startX
        else:
            data_spaced_x = np.power(10,(data_spaced_x/width*(np.log10(self.endX)-np.log10(self.startX))+np.log10(self.startX)))
        if(self.yaxis_type == AxisType.LINEAR):
            data_spaced_y = data_spaced_y/height*(self.endY-self.startY)+self.startY
        else:
            data_spaced_y = np.power(10,(data_spaced_y/height*(np.log10(self.endY)-np.log10(self.startY))+np.log10(self.startY)))
        return np.array([data_spaced_x,data_spaced_y]).T
    
    def change_eraser(self):
        self.erase_range = self.horizontalSlider_eraser.value()
        self.update_cursor(ICON_ERASER,self.erase_range*2)
        
    def color_picker(self):
        self.label_img.mousePressEvent = self.pick_color
        self.pushButton_preview.setEnabled(True)
        self.update_cursor(ICON_PICKER,40)
        self.system_state = SystemState.PICKING_COLOR
        self.label_info.setText("Pick color")
        
        
    def update_info(self,msg):
        self.label_info.setText(msg)

    def eraser(self):
        self.update_info("erase noise")
        self.change_eraser()
        self.system_state = SystemState.ERASING

    def color_extractor(self):
        self.read_config()
        [x1,y2] = self.pos_left_bottom
        [x2,y1] = self.pos_right_top
        if y1 != y2 and x1 != x2 and len(self.mask):
            tailored_mask = self.mask[y1:y2,x1:x2]
            extracted_data = self.extract_data(tailored_mask)
            # 数据点映射坐标
            mapped_data = self.data_mapping(extracted_data,tailored_mask.shape[1],tailored_mask.shape[0])
        else:
            QMessageBox.warning(self,"Warning",'1. Make sure you have selected the frame of the graph\n2. Make sure you have use picker to select the graph you want to extract')
            return None
        # self.add_curve_to_list(mapped_data)
        return mapped_data

    def add_curve_to_list(self,data,name):
        curve_idx = name
        if curve_idx not in self.result_list.keys():
            self.result_list[curve_idx] = data
        else:
            ans = QMessageBox.question(self, "Info", "The name of the curve has existed\nDo you want to overwrite it?", QMessageBox.Yes | QMessageBox.No)
            if(ans == QMessageBox.Yes):
                self.result_list[curve_idx] = data
            
        self.update_info('Curve added')

    def update_cursor(self,filepath,size=40):
        """Update cursor style

        Args:
            filepath (str): cursor image path
            size (int): size of cursor, defaults to 40
        """

        cursor_qmap = QtGui.QPixmap(filepath)
        cursor_qmap_scaled = cursor_qmap.scaled(QSize(size,size),Qt.KeepAspectRatio)
        self.label_img.setCursor(QtGui.QCursor(cursor_qmap_scaled,-1,-1))

    def export_data(self):
        self.last_path = self.setting.value('LastFilePath')
        if(self.last_path is None):
            self.last_path = '/'
        filename=QFileDialog.getSaveFileName(self,'save file',filter="Txt files(*.txt)",directory=self.last_path)[0]
        print(dir)
        self.setting.setValue('LastFilePath', os.path.dirname(filename))
        if(filename == ''):
            return
        for key,value in self.result_list.items():
            # print(value)
            filename = f"{filename[:-4]}_{key}.txt"
            print(filename)
            savetxt(f"{filename}",value,delimiter=';')

    def export_data_csv(self):
        self.last_path = self.setting.value('LastFilePath')
        if(self.last_path is None):
            self.last_path = '/'

        self.setting.setValue('LastFilePath', os.path.dirname(filename))
        filename=QFileDialog.getSaveFileName(self,'save file',filter="Data files(*.csv)")[0]
        if(filename == ''):
            return
        for key,value in self.result_list.items():
            # print(value)
            filename = f"{filename[:-4]}_{key}.csv"
            print(filename)
            savetxt(f"{filename}",value,delimiter=',')

    def import_img(self):
        # try:
        filename=QFileDialog.getOpenFileName(self,'open file')[0]
        # print(filename)
        if not os.path.exists(filename):
            return
        self.load_img_from_file(file = filename)

    def auto_mode(self):
        [x,y,w,h] = find_box(self.current_img)
        box_x1 = x
        box_x2 = x+w
        box_y1 = y
        box_y2 = y+h
        self.pos_left_bottom = [box_x1,box_y2]
        self.pos_right_top = [box_x2,box_y1]
        self.has_frame = True
        self.refresh_img()

    def keyboardEventReceived(self, event):
        if event.event_type == 'down':
            if event.name == '[':
                current_value = self.horizontalSlider_eraser.value()
                self.horizontalSlider_eraser.setProperty("value", current_value-1)
                print('[ pressed')
            elif event.name == ']':
                current_value = self.horizontalSlider_eraser.value()
                self.horizontalSlider_eraser.setProperty("value", current_value+1)
                print('] pressed')

    def add_curve(self):
        curve_name,done = QtWidgets.QInputDialog.getText(self, 'Info', "Input curve name")
        if(done):
            self.curve_idx+=1
            data = self.color_extractor()
            if(data is not None):
                self.add_curve_to_list(data,curve_name)

    def plot_value(self):
        plt.figure(figsize=(5,5))
        for data in self.result_list.values():
            plt.plot(data[:,0],data[:,1],"-",linewidth=2)

        if(self.xaxis_type == AxisType.LOG):
            plt.xscale("log")
        if(self.yaxis_type == AxisType.LOG):
            plt.yscale("log")
        plt.rcParams['font.sans-serif']=['SimHei']###解决中文乱码
        plt.rcParams['axes.unicode_minus']=False
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Extract Data")
        plt.grid(True,which='both',ls='--')
        plt.legend(self.result_list.keys())
        plt.show()

