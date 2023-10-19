import argparse
import time
import cv2
import numpy as np
import math

def cv_show(name,img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


step1 = 1
step2 = 1

def getDistance(source, template):  # calculate the matching degree

	sum = 0

	temp = source - template

	YY, XX, CC = source.shape[0:3]
	for y in range(0,YY,step1):
		for x in range(0,XX,step1):
			for c in range(0,CC):
				sum = sum + temp[y,x,c] ** 2
	return math.sqrt(sum)

def myMatchTemplate(source, template):
	hs, ws = source.shape[:2]    # height and width of the source image
	ht, wt = template.shape[:2]  # height and width of the template
	dist, rx,ry = 1000000,0,0

	for x in range(0, ws-wt, step2 ):       # slide the template -x
		for y in range(0, hs-ht, step2 ):   # slide the template -y

			temp = getDistance(  source[y:y + ht, x:x + wt, :], template )  # calculate the matching degree
			if temp < dist:
				dist = temp
				rx = x
				ry = y

	return rx,ry


#vs = cv2.VideoCapture("soccer_01.mp4")
vs = cv2.VideoCapture("los_angeles.mp4")

template = {}
template[0] = False

while True:
	# 取当前帧
	frame = vs.read()

	frame = frame[1] # (true, data)
	if frame is None:
		break

	(h, w) = frame.shape[:2]
	width=600
	r = width / float(w)
	dim = (width, int(h * r))
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


	if template[0] != False: #
	#	rx,ry = myMatchTemplate(frame,template[1])
		res = cv2.matchTemplate(frame,template[1],cv2.TM_SQDIFF)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

		top_left = min_loc                                                  # the top left
		bottom_right = ( top_left[0]+template[2], top_left[1]+template[3])  # the bottom_right
		cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 2)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(100) & 0xFF

	if key == ord("s"):   # select an object
		box = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
		print(box)

        # to save the object(pixels) as the template
		tx,ty,tw,th = box[0:4]
		template[0] = True
		template[1] = np.copy(frame[ty:ty+th, tx:tx+tw])  # deep copy
		template[2] = tw
		template[3] = th

	elif key == 27:
		break


vs.release()
cv2.destroyAllWindows()