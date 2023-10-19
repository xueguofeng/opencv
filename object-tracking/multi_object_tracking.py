import argparse
import time
import cv2
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",help="OpenCV object tracker type")
args = vars(ap.parse_args())


# The tracking algorithms that Opencv supported
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.legacy.TrackerKCF_create,
	"boosting": cv2.legacy.TrackerBoosting_create,
	"mil": cv2.legacy.TrackerMIL_create,
	"tld": cv2.legacy.TrackerTLD_create,
#	"medianflow": cv2.TrackerMedianFlow_create,
#	"mosse": cv2.TrackerMOSSE_create
}


# create an OpenCV's multi-object tracker
trackers = cv2.legacy.MultiTracker_create()

vs = cv2.VideoCapture(args["video"])

while True:

	frame = vs.read()
	frame = frame[1] 	# (true, data)

	if frame is None:
		break

	# resize
	(h, w) = frame.shape[:2]
	width=600
	r = width / float(w)
	dim = (width, int(h * r))
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

	# get the latest update of objects tracked by Opencv
	(success, boxes) = trackers.update(frame) # the current frame

	# every box is from a new frame, box position should change often
	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(100) & 0xFF

	if key == ord("s"):
		# Press 's' to select a region and then press any key to continue
		box = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)

		tracker = OPENCV_OBJECT_TRACKERS[ args["tracker"] ]()
		trackers.add(tracker, frame, box)
	    # add a specific tracker: 'cv2.legacy.TrackerKCF_create', the current frame, the selected box

	elif key == 27: 	# ESC to exit
		break


vs.release()
cv2.destroyAllWindows()