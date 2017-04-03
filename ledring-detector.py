#!/usr/bin/env python

import cv2
import numpy as np

import sys

if len(sys.argv) < 2:
    print("Usage: {} <image>".format(sys.argv[0]))
    sys.exit(1)

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
cv2.imshow('source', img)

ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('threshold', img)

# kernel = np.ones((2, 2), np.uint8)
# img = cv2.erode(img, kernel)
# cv2.imshow('erode', img)
# cv2.dilate(img, kernel, iterations=1)

# Detecting blob
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 1
params.maxArea = 1000

params.filterByInertia = False
params.filterByArea = False
# params.filterByColor = False
params.filterByConvexity = False

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(cv2.bitwise_not(img))
imagePoints = []
for point in keypoints:
    imagePoints.append([point.pt[0], point.pt[1]])
imagePoints = np.array(imagePoints)
print(imagePoints)
print(cv2.convexHull(imagePoints))
img = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('keypoints', img)

# SolvePnP
if len(imagePoints) == 4:
    w = 0.02
    objectPoints = [[0, w, 0],
                    [-w, 0, 0],
                    [w, 0, 0],
                    [0, -w, 0]]
    objectPoints = np.array(objectPoints)
    cameraMatrix = np.array([[6.2321787645563745e+02, 0., 3.2242878461613770e+02],
                             [0., 6.2360826877593502e+02, 2.4359896236415980e+02],
                             [0., 0., 1.]])
    distCoeffs = np.array([[1.1142910472083294e-01], [-2.6868764924280603e-01],
                          [1.6666109149621208e-03], [2.3268549597284692e-03],
                          [2.8967322731960432e-01]])

    res, r, t = cv2.solvePnP(objectPoints, imagePoints,
                             cameraMatrix, distCoeffs)

    if res:
        print("Position: {}, {}, {}".format(t[0], t[1], t[2]))
    else:
        print("SolvePnP failed!")
else:
    print("Cannot detect enough (4) points")

cv2.waitKey(0)
cv2.destroyAllWindows()
