import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import sys
import cv2
import glob
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.acceptDrops()
        #widget = QWidget()
        #path = glob.glob("C:/Users/pawel/Desktop/yaleB11/*.pgm")
        self.cv_img = []
        for img in images:
            n = cv2.imread(img)
            height, width, channel = n.shape
            bytesPerLine = 3 * width
            qn = QImage(n.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.cv_img.append(qn)

        self.i = 0
        self.setWindowTitle("Face recognition")
        self.label = QLabel(self)
        self.setGeometry(0, 0, 640, 600)
        self.pixmap = QPixmap(self.cv_img[self.i])
        self.label.setPixmap(self.pixmap)
        self.label.resize(self.pixmap.width(),
                          self.pixmap.height())
        self.button_next = QPushButton("Next", self)
        self.button_previous = QPushButton("Previous", self)

        self.button_next.clicked.connect(self.button1_clicked)

        self.button_next.setGeometry(340, 500, 80, 30)

        self.button_previous.clicked.connect(self.button2_clicked)
        self.button_previous.setGeometry(220, 500, 80, 30)

        self.show()

    def button1_clicked(self):
        self.i = self.i + 1
        tempPix = QPixmap(self.cv_img[self.i])
        self.label.setPixmap(tempPix)

    def button2_clicked(self):
        if self.i > 0:
            self.i = self.i - 1

        tempPix = QPixmap(self.cv_img[self.i])
        self.label.setPixmap(tempPix)



