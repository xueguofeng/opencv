import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#vid = cv2.VideoCapture('traffic.avi')
vid = cv2.VideoCapture('test.avi')
#vid = cv2.VideoCapture(0)
ret, frame = vid.read()


h,w = frame.shape[:2]


alpha   = 0.10              # the contribution of new pixel is the existing model
stdInit = 20                # standard deviation
varInit = stdInit * stdInit # variance
lamda   = 2.5 * 1.2         # the threshold


frame_u = np.zeros( shape = frame.shape )
frame_d = np.zeros( shape = frame.shape )
frame_std = np.zeros( shape= frame.shape )
frame_var = np.zeros( shape = frame.shape )


# Use the first image to initialize the 4 matrices
for i in range(h):
    for j in range(w):

        # current pixel
        pixel_R = frame[i, j, 0]
        pixel_G = frame[i, j, 1]
        pixel_B = frame[i, j, 2]

        # Background - Expectation
        pixel_ur = pixel_R
        pixel_ug = pixel_G
        pixel_ub = pixel_B

        # Foreground
        pixel_dr = 0
        pixel_dg = 0
        pixel_db = 0

        # Standard Deviation
        pixel_stdr = stdInit
        pixel_stdg = stdInit
        pixel_stdb = stdInit

        # Variance
        pixel_varr = varInit
        pixel_varg = varInit
        pixel_varb = varInit

        frame_u[i, j, :]   = [pixel_ur, pixel_ug, pixel_ub]  # Background - Expectation, for each pixel
        frame_d[i, j, :]   = [pixel_dr, pixel_dg, pixel_db] # Foreground, for each pixel
        frame_std[i, j, :]  = [pixel_stdr, pixel_stdg, pixel_stdb] # Standard Deviation, for each pixel
        frame_var[i, j,:] = [pixel_varr, pixel_varg, pixel_varb] # Variance,for each pixel

'''
cv_show('frame',frame)
cv_show('frame_u',frame_u.astype(np.uint8))
cv_show('frame_d',frame_d.astype(np.uint8))
cv_show('frame_std',frame_std.astype(np.uint8))
cv_show('frame_var',frame_var.astype(np.uint8))
'''

while True:
    ret, frame = vid.read()
    if frame is None:
        break
    else:

        for i in range(h):
            for j in range(w):

                # current pixel in the new frame
                pixel_R = frame[i, j, 0]
                pixel_G = frame[i, j, 1]
                pixel_B = frame[i, j, 2]

                # get the Background - Expectation
                pixel_ur = frame_u[i, j, 0]
                pixel_ug = frame_u[i, j, 1]
                pixel_ub = frame_u[i, j, 2]

                # get the Standard Deviation
                pixel_stdr = frame_std[i, j, 0]
                pixel_stdg = frame_std[i, j, 1]
                pixel_stdb = frame_std[i, j, 2]

                # get the Variance
                pixel_varr = frame_var[i, j, 0]
                pixel_varg = frame_var[i, j, 1]
                pixel_varb = frame_var[i, j, 2]

                # the pixel in the new image, is the background when | I - u | < lamda * std
                pixel_dr = pixel_R - pixel_ur
                pixel_dg = pixel_G - pixel_ug
                pixel_db = pixel_B - pixel_ub



                Need = (abs(pixel_dr) < (lamda * pixel_stdr)) and (abs(pixel_dg) < (lamda * pixel_stdg)) and (abs(pixel_db) < (lamda * pixel_stdb))

                if Need == True: # the pixel in the new image is the background

                    # the new Background - Expectation, u = (1 - alpha) * u + alpha * 1
                    pixel_ur = (1 - alpha) * pixel_ur + alpha * pixel_R
                    pixel_ug = (1 - alpha) * pixel_ug + alpha * pixel_G
                    pixel_ub = (1 - alpha) * pixel_ub + alpha * pixel_B

                    # the new Variance, var = (1 - alpha) * var + alpha * (I - u) ^ 2
                    pixel_varr = (1 - alpha) * pixel_varr + alpha * (pixel_R - pixel_ur) ** 2
                    pixel_varg = (1 - alpha) * pixel_varg + alpha * (pixel_G - pixel_ug) ** 2
                    pixel_varb = (1 - alpha) * pixel_varb + alpha * (pixel_B - pixel_ub) ** 2

                    # the new Standard Deviation
                    pixel_stdr = math.sqrt(pixel_varr)
                    pixel_stdg = math.sqrt(pixel_varg)
                    pixel_stdb = math.sqrt(pixel_varb)

                    # update
                    frame_u[i, j, :] = [pixel_ur, pixel_ug, pixel_ub]
                    frame_std[i, j, :] = [pixel_stdr, pixel_stdg,pixel_stdb]
                    frame_var[i, j, :] = [pixel_varr, pixel_varg, pixel_varb]

                    frame_d[i, j, :] = [0, 0, 0]


                else:          # the pixel in the new image is the foreground

                    frame_d[i, j, :] = [pixel_dr, pixel_dg, pixel_db]





    cv2.imshow('frame',frame)
    cv2.imshow('frame_d', frame_d)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

vid.release()
cv2.destroyAllWindows()