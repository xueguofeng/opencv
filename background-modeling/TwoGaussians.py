import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

vid = cv2.VideoCapture('traffic.avi')

frames = []
frame_count = 0

while True:
    ret, frame = vid.read()
    if frame is not None:
        #frame = cv2.resize(frame,(50,50))
        frames.append(frame)
        frame_count += 1
    else:
        break
frames = np.array(frames)
frames = frames[:,:,:,:]

print("Number of frames extracted is {}".format(frame_count))

print("array dimensions will be (num_frames, image_width, image_height, num_channels)")
print("Shape of frames is {}".format(frames.shape))



gmm = GaussianMixture(n_components = 2)

# initialize a dummy background image with all zeros
background = np.zeros(shape=(frames.shape[1:]))


print("Shape of dummy background image is {}".format(background.shape))



for i in range(frames.shape[1]):
    for j in range(frames.shape[2]):
        for k in range(frames.shape[3]):
            X = frames[:, i, j, k]
            X = X.reshape(X.shape[0], 1)
            gmm.fit(X)
            means = gmm.means_
            covars = gmm.covariances_
            weights = gmm.weights_
            idx = np.argmax(weights)  # index of the biggest weight
            background[i][j][k] = int(means[idx])

    print(i,j)

# Store the result onto disc
cv2.imwrite('background.png', background)

