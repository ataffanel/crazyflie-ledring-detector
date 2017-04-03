#!/usr/bin/env python

import cv2
import numpy as np

import sys

import zmq

if len(sys.argv) < 2:
    print("Usage: {} <camera_id>".format(sys.argv[0]))
    sys.exit(1)

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:7777")

cap = cv2.VideoCapture(int(sys.argv[1]))

while True:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('source', img)

    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', img)

    # Detecting blob
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 1000

    params.filterByInertia = False
    params.filterByArea = False
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(cv2.bitwise_not(img))
    imagePoints = []
    for point in keypoints:
        imagePoints.append([point.pt[0], point.pt[1]])
    imagePoints = np.array(imagePoints)

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
