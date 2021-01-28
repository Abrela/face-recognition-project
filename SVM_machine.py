#Project EIASR

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score
from sklearn import svm
import sys
import pickle
from pathlib import Path
import detect_face


# setting paths
from PyQt5.QtWidgets import QApplication

def learn_and_check():
    #folderpath = "D:/ExtendedYaleB/"
    folderpath = "D:/CroppedYale/"
    height = 256
    width = 128
    data = []
    labels = []
    gray = []
    if not Path('linear_model.sav').is_file() and not Path('rbf_model.sav').is_file():
        for dirname, _, filenames in tqdm(os.walk(folderpath)):
            for filename in filenames:
                if filename.endswith('.pgm'):
                    image = cv2.imread(os.path.join(dirname, filename))
                    image = cv2.resize(image, (width, height))
                    labels.append(dirname.split("/").pop())
                    data.append(image)
                    print(filename)

        #for i in range(len(data)):
        #    gray.append(detect_face.face_crop(data[i]))
        #    print(i)

        hog_features = []
        hog_image = []
        for image in tqdm(data):

            fd, hogim = hog(image, orientations=9, pixels_per_cell=(8, 8),
                            block_norm='L2', cells_per_block=(2, 2), visualize=True)
            hog_image.append(hogim)
            hog_features.append(fd)

        #Preparing input data to SVM model

        Labels = np.array(labels)#.reshape(len(labels), 1) #labels into stack of arrays
        hog_features = np.array(hog_features)

        #data_frame = np.hstack((hog_features, Labels))
        #np.random.shuffle(data_frame)

        print(hog_features.size)
        print(Labels.size)
        #x_train, x_test - the training and test part of the first sequence data_frame[:,:-1]
        #y_train, y_test - the training and test part of the second sequence data_frame[:,-1]
        x_train, x_test, y_train, y_test = train_test_split(hog_features,
                                                            Labels,
                                                            test_size=0.3,
                                                            random_state=42
                                                            )

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



        linear_model.fit(x_train_lin, y_train_lin)
        rbf_model.fit(x_train_rbf, y_train_rbf)
        pickle.dump(linear_model, open('linear_model.sav', 'wb'))
        pickle.dump(rbf_model, open('rbf_model.sav', 'wb'))
    else:
        linear_model = pickle.load(open('linear_model.sav', 'rb'))
        rbf_model = pickle.load(open('rbf_model.sav', 'rb'))
    return linear_model, rbf_model


def load_picture_to_recognize(path, classifier):
    print(classifier)
    fd, hogim = hog(path, orientations=9, pixels_per_cell=(8, 8),
                    block_norm='L2', cells_per_block=(2, 2), visualize=True)

    print(fd.size)

    fd2 = [fd]
    y_pred = classifier.predict(fd2)
    print("load")

    return y_pred
#RESULT

#print('\n')
#print('LINEAR SVM MODEL')
#print('\n')
#print("Accuracy: " + str(accuracy_score(y_test_rbf, y_pred_rbf)))
#print('\n')
#print(classification_report(y_test_rbf, y_pred_rbf))
#print('\n')
#print('RBF SVM MODEL')
#print('\n')
#print("Accuracy: " + str(accuracy_score(y_test_lin, y_pred_lin)))
#print('\n')
#print(classification_report(y_test_lin, y_pred_lin))
