import numpy as np
import cv2

def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


cap = cv2.VideoCapture('test.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

fgbg = cv2.createBackgroundSubtractorMOG2()

while(True):
    ret, frame = cap.read()

    fg_mask = fgbg.apply(frame)

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

#    cv_show('original',frame)
#    cv_show('temp', fg_mask)

    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:

        perimeter = cv2.arcLength(c,True)
        if perimeter > 188:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('frame',frame)
    cv2.imshow('fgmask', fg_mask)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()