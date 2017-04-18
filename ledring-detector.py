#!/usr/bin/env python

import cv2
import numpy as np
import opencvloader

import sys

import zmq

if len(sys.argv) < 3:
    print("Usage: {} <camera_calib.yml> <camera_id>".format(sys.argv[0]))
    sys.exit(1)

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:7777")

cameraParameters = opencvloader.loadYaml(sys.argv[1])

cap = cv2.VideoCapture(int(sys.argv[2]))

while True:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('source', img)

    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', cv2.bitwise_not(img))

    # Detecting blob
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 1000

    params.filterByInertia = False
    params.filterByArea = False
    params.filterByConvexity = False

    params.minDistBetweenBlobs = 0

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(cv2.bitwise_not(img))
    imagePoints = []
    for point in keypoints:
        imagePoints.append([point.pt[0], point.pt[1]])
    imagePoints = np.array(imagePoints).astype(np.float32)

    img = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('keypoints', img)

    # SolvePnP
    if len(imagePoints) == 4:
        # Find convex hull of the points (we ignore orientation)
        imagePoints = cv2.convexHull(imagePoints.reshape(4, 1, 2))
        imagePoints = imagePoints.reshape(4, 2)

        w = 0.026/2.0
        objectPoints = [[w, 0, 0],
                        [0, w, 0],
                        [-w, 0, 0],
                        [0, -w, 0]]
        objectPoints = np.array(objectPoints)
        cameraMatrix = cameraParameters["camera_matrix"]
        distCoeffs = cameraParameters["distortion_coefficients"]

        res, r, t = cv2.solvePnP(objectPoints, imagePoints,
                                 cameraMatrix, distCoeffs)

        if res:
            print("Position: {}, {}, {}".format(t[0][0], t[1][0], t[2][0]))
            msg = {"pos": [t[0][0], t[1][0], t[2][0]], "detect": True}
            socket.send_json(msg)
        else:
            print("SolvePnP failed!")
            msg = {"pos": [0, 0, 0], "detect": False}
            socket.send_json(msg)
    else:
        print("Cannot detect enough (4) points")
        msg = {"pos": [0, 0, 0], "detect": False}
        socket.send_json(msg)

    key = cv2.waitKey(1)
    if key != 255:
        break

cv2.destroyAllWindows()
