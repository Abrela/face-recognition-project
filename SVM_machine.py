#Project EIASR

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score
from sklearn import svm
import sys, Window
import pickle
from pathlib import Path
import detect_face, Window
import hog_function

# setting paths
from PyQt5.QtWidgets import QApplication

folderpath = "C:/Users/brela/Desktop/Pw_3_Semester/EIASR/Images/"
cascade = "C:/Users/brela/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml"

height = 156
width = 128
data = []
labels = []

for dirname, _, filenames in tqdm(os.walk(folderpath)):
    for filename in filenames:
        image = cv2.imread(os.path.join(dirname, filename))
        image = cv2.resize(image, (width,height))
        labels.append(dirname.split("/").pop())
        data.append(image)

fig = plt.figure(figsize=(10,10))
for i in range(1,5):
    index = random.randint(0,2925)
    plt.subplot(2,2,i)
    plt.imshow(data[index])
    plt.xlabel(labels[index])
plt.show()

gray = [detect_face.face_crop(data[i])for i in range(len(data))]


fig = plt.figure(figsize=(10,10))
for i in range(1,5):
    index = random.randint(0,2925)
    plt.subplot(2,2,i)
    plt.imshow(gray[index])
    plt.xlabel(labels[index])
plt.show()

hog_features=[]
hog_image=[]
for image in tqdm(gray):
    fd = hog_function.get_hog_vector(image)
    #fd , hogim = hog(image , orientations=9 , pixels_per_cell=(16,16) , block_norm='L2' , cells_per_block=(4,4) , visualize=True)
    #hog_image.append(hogim)
    hog_features.append(fd)

fig = plt.figure(figsize=(10,10))
for i in range(1,5):
    index = random.randint(0,2925)
    plt.subplot(2,2,i)
    plt.imshow(hog_image[index])
    plt.xlabel(labels[index])
plt.show()


#Preparing input data to SVM model

Labels = np.array(labels).reshape(len(labels),1) #labels into stack of arrays
hog_features = np.array(hog_features)
data_frame = np.hstack((hog_features,Labels))
np.random.shuffle(data_frame)

#x_train, x_test - the training and test part of the first sequence data_frame[:,:-1]
#y_train, y_test - the training and test part of the second sequence data_frame[:,-1]
x_train , x_test , y_train , y_test = train_test_split(data_frame[:,:-1] ,
                                                       data_frame[:,-1],
                                                       test_size=0.3 ,
                                                       random_state=42 ,
                                                       stratify=data_frame[:,-1])

x_train_lin = x_train
x_train_rbf = x_train
x_test_lin = x_test
x_test_rbf = x_test
y_train_lin = y_train
y_train_rbf = y_train
y_test_lin = y_test
y_test_rbf = y_test

#Create SVM model to fit

linear_model = svm.SVC(kernel='linear' , class_weight='balanced' , C=10 , gamma='scale')
rbf_model = svm.SVC(kernel='rbf' , class_weight='balanced' , C=10 , gamma='scale')


if not Path('linear_model.sav').is_file() and not Path('rbf_model.sav').is_file():
    linear_model.fit(x_train_lin, y_train_lin)
    rbf_model.fit(x_train_rbf, y_train_rbf)
    pickle.dump(linear_model, open('linear_model.sav', 'wb'))
    pickle.dump(rbf_model, open('rbf_model.sav', 'wb'))
else:
    linear_model = pickle.load(open('linear_model.sav', 'rb'))
    rbf_model = pickle.load(open('rbf_model.sav', 'rb'))

y_pred_lin = linear_model.predict(x_test_lin)
y_pred_rbf = rbf_model.predict(x_test_rbf)

#RESULT

print('\n')
print('LINEAR SVM MODEL')
print('\n')
print("Accuracy: " + str(accuracy_score(y_test_rbf, y_pred_rbf)))
print('\n')
print(classification_report(y_test_rbf, y_pred_rbf))
print('\n')
print('RBF SVM MODEL')
print('\n')
print("Accuracy: " + str(accuracy_score(y_test_lin, y_pred_lin)))
print('\n')
print(classification_report(y_test_lin, y_pred_lin))

App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# start the app
sys.exit(App.exec())