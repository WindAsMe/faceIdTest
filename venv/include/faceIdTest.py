# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :18-5-17 下午3:09
# File     :faceIdTest.py
# Location:/Home/PycharmProjects/..

# Face detected


import cv2
import numpy as np


def face_detected():
    # Initial the classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    # If classifier not found, raise error
    if face_cascade.empty():
        raise IOError("Not Found !")

    # Open the video
    cap = cv2.VideoCapture(0)

    # Define the coefficient
    scaling_factor = 0.5
    while True:
        # Collecting current frame
        # DO PRETREATMENT
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # Gray the pic
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Running the cascade
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw the rectangle in face
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

        # Show the pic
        cv2.imshow('Face Detector', frame)

        # Check 'Esc' is pressed
        c = cv2.waitKey(1)
        if c == 27:
            break

    # Release the resource
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_detected()