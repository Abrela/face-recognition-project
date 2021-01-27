# TODO #1: Reimplementing functions and methods
# TODO #2: Automatization of whole process

# detecting faces in loaded images
# prototype works sufficiently with pictures with only one face prsent

#importing libraries
import cv2
import os
import numpy as np
from PIL import Image

def face_crop(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    #path_write = 'C:/Users/Sebastian/OneDrive - Politechnika Warszawska/Pulpit/EIASR_Project/result'
    #path_read = 'C:/Users/Sebastian/OneDrive - Politechnika Warszawska/Pulpit/EIASR_Project/try'

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray is None:
        raise Exception("No image")


    # normalizing image
    normalizedImg = np.zeros((800, 800))
    #normalizedImg = np.zeros((5, 2))
    result = cv2.normalize(gray,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    # result = cv2.normalize(gray,  normalizedImg, 0, 255, cv2.NORM_L1)
    im = Image.fromarray(result)
    #cv2.imshow('Normalized',result)
    #cv2.imwrite(os.path.join(path_write, 'Normalized.pgm'), gray )

    # face cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # drawing rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (204,0,204), 4)

    # cropping image
    if len(faces) == 1:
        crop_img = gray[y:y+h, x:x+w]
        return crop_img
    else:
        return gray
