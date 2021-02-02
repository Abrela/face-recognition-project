import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import sys
import cv2
import glob
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from SVM_machine import learn_and_check, load_picture_to_recognize
import detect_face
import numpy as np

class Window(QDialog):
    def __init__(self):
        super().__init__()
        #self.acceptDrops()
        #widget = QWidget()
        linear_model, rbf_model = learn_and_check()
        self.classifierLin = linear_model
        self.classifierRbf = rbf_model

        #self.pixmap = QPixmap(self.cv_img[self.i])
        #self.label.setPixmap(self.pixmap)
        #self.label.resize(self.pixmap.width(),
        #                  self.pixmap.height())
        #self.button_next = QPushButton("Next", self)
        #self.button_previous = QPushButton("Previous", self)

        #self.button_next.clicked.connect(self.button1_clicked)

        #self.button_next.setGeometry(340, 500, 80, 30)

       # self.button_previous.clicked.connect(self.button2_clicked)
        #self.button_previous.setGeometry(220, 500, 80, 30)
        self.InitUI()


    def InitUI(self):
        self.setWindowTitle("Face recognition")
        self.setGeometry(320, 300, 640, 600)
        image_box = QVBoxLayout()
        self.path_button = QPushButton("Choose image")
        image_box.addWidget(self.path_button)
        self.label = QLabel("")
        image_box.addWidget(self.label)
        self.path_button.clicked.connect(self.getImage)
        self.setLayout(image_box)
        self.face_name = QLabel("Podpis")
        image_box.addWidget(self.face_name)
        self.face_name.move(600,600)
        self.show()

    def getImage(self):
        dialog = QFileDialog.getOpenFileName(self, 'Choose', 'D:', 'Image files (*.pgm)')
        img = dialog[0]
        pixmap = QPixmap(img)
        self.label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())
        img = cv2.imread(img)
        #ans = detect_face.face_crop(img)
        ret = load_picture_to_recognize(img, self.classifierLin)
        print("load")
        self.face_name.setText(np.array_str(ret))
        print(ret)

    def button1_clicked(self):
        self.i = self.i + 1
        tempPix = QPixmap(self.cv_img[self.i])
        self.label.setPixmap(tempPix)

    def button2_clicked(self):
        if self.i > 0:
            self.i = self.i - 1

        tempPix = QPixmap(self.cv_img[self.i])
        self.label.setPixmap(tempPix)



App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# start the app
sys.exit(App.exec())