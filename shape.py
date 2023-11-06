#!/usr/bin/env/python3
import numpy as np
import matplotlib as plt
import math
import cv2

img = cv2.imread("vllCR.png",0)
img1 = cv2.imread("vllCR.png",1)
hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
ret,thres = cv2.threshold(img,240,255,cv2.THRESH_BINARY) #white black no gray
contours,_ = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


for i in contours:
    #remove imperfections
    epsilon = 0.01*cv2.arcLength(i,True)
    approx = cv2.approxPolyDP(i,epsilon,True)
    cv2.drawContours(img,[approx],0,(0),5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    
    if len(approx) == 3:
        cv2.putText(img1,'TRIANGLE',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0))
    elif len(approx) == 5:
        cv2.putText(img1,'PENTAGON',(approx.ravel()[2],approx.ravel()[3]),cv2.FONT_HERSHEY_SIMPLEX,1,(0))
    elif(len(approx) == 4):
        x,y,w,h = cv2.boundingRect(approx)
        a_r = w/h
        if(a_r>0.95 and a_r<1.05):
            cv2.putText(img1,'SQUARE',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0))
        else:
            cv2.putText(img1,'Rectangle',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0))
    elif len(approx) > 14 and len(approx)<17:
        cv2.putText(img1,"circle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
    if len(approx)>10 and len(approx)<14:
        cv2.putText(img1,"ellipse",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))




cv2.imshow("window2",thres)
cv2.imshow("window",img)
cv2.imshow("prachit",img1)
cv2.imshow("prat",mask_yellow)
k = cv2.waitKey(0)
if k == ord('z'):
    cv2.destroyAllWindows()
elif k ==ord('s'):
    cv2.imwrite("shapes_detected.png",img1)
