import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

vid = cv2.VideoCapture('traffic.avi')

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

background = cv2.imread("background.png")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

vid = cv2.VideoCapture('traffic.avi')
while True:
    ret, frame = vid.read()
    if frame is not None:

        diff = frame - background

        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

        cv2.imshow('foreground',diff)

        k = cv2.waitKey(150) & 0xff
        if k == 27:
            break

    else:
        break