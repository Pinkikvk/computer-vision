#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 17:29:06 2019

@author: jakubkrawczyk
"""

import cv2
from smile_detector import SmileDetector

def resize(img):
    return cv2.resize(img, (640, 360), interpolation = cv2.INTER_AREA)

cap = cv2.VideoCapture(0)
sd = SmileDetector()

while(True):
    ret, frame = cap.read()
    img = resize(frame)
    smile_detected = sd.detect(img)
    
    if smile_detected:
        cv2.putText(img, "Smile Detected", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(img, "Smile Not Detected", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Smile Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()