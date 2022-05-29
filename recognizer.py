#!/usr/bin/env python3.10

import cv2
import numpy as np

from datetime import datetime

import time

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./model/trained_model2.yml')
cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

while True:
    #cam = cv2.VideoCapture(0)
    ret, im = cam.read()
    if ret is None or im is None:
        print(datetime.now(), ret, im)
        time.sleep(0.5)
        continue
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for(x, y, w, h) in faces:
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        cv2.rectangle(im, (x, y), (x+w, y+h),
                (0, 260, 0), 7)
        cv2.putText(im, str(Id), (x, y-40),
                font, 2, (255, 255, 255), 3)

    cv2.imshow('im', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
