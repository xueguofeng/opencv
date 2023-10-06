from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import argparse
import time
import dlib
import cv2


FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])


# http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
# 6 points, from 0 to 5
def eye_aspect_ratio(eye):
	# Vertical Distance
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# Horizontal Distance
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C) # EAR Value
	return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())


EYE_AR_THRESH = 0.3 # eyes are close
EYE_AR_CONSEC_FRAMES = 3  # eyes are close by 3 frames

COUNTER = 0
TOTAL = 0

# 检测与定位工具
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# two eyes
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]



print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(args["video"])
#vs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

def shape_to_np(shape, dtype="int"):
	# return 68 points, (x,y)
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords


while True:

	frame = vs.read()[1]
	if frame is None:
		break
	
	(h, w) = frame.shape[:2]
	width = 1200
	r = width / float(w)
	dim = (width, int(h * r))
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Find faces first
	rects = detector(gray, 0)

	# traverse each face
	for rect in rects:

		shape = predictor(gray, rect)
		shape = shape_to_np(shape) # get 68 points by calling shape's member functions

		# 2 EAR Values
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# AVERAGE
		ear = (leftEAR + rightEAR) / 2.0

		# Draw eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# eyes are close
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		else:
			# eyes are close for 3 frames
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			# reset
			COUNTER = 0

		# show
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(2) & 0xFF

	if key == 27:
		break

vs.release()
cv2.destroyAllWindows()