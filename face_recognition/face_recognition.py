#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 17:29:06 2019

@author: jakubkrawczyk
"""

import cv2

def resize(img):
    return cv2.resize(img, (640, 360), interpolation = cv2.INTER_AREA)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray_img, original_image):
    faces = face_cascade.detectMultiScale(gray_img,  1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(original_image, (x, y), (x+w, y+h), (255,0,0), 1)
        face_gray_img = gray_img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray_img, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(original_image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 1)
    return original_image


cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    img = resize(frame)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = detect(gray_img, img)
    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()