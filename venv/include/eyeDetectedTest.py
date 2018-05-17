# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :18-5-17 下午7:19
# File     :eyeDetectedTest.py
# Location:/Home/PycharmProjects/..

# Eye and nose detected

import cv2
import numpy as np


def eye_detected():
    # Initial the classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # If classifier not found, raise error
    if face_cascade.empty():
        raise IOError('Face cascade not found !')
    if eye_cascade.empty():
        raise IOError('Eye cascade not found !')

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
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Eye is in the face
        for (x, y, w, h) in faces:
            # Acquire the face ROI information
            # From grey pic and pic
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect the eye in gray
            eye_rects = eye_cascade.detectMultiScale(roi_gray)

            # Draw green circle around eye
            for (x_eye, y_eye, w_eye, h_eye) in eye_rects:
                center = (int(x_eye + 0.5 * w_eye), int(y_eye + 0.5 * h_eye))
                radius = int(0.3 * (w_eye + h_eye))
                color = (0, 255 ,0)
                thickness = 3
                cv2.circle(roi_color, center, radius, color, thickness)

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
    eye_detected()