#!/usr/bin/env python3
from pyzbar.pyzbar import decode
import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)

def get_qr_code(inframe):
    try:
        return(decode(inframe))
    except:
        return([])

def draw_polygon(inframe,qrobj):
    if len(qrobj) == 0:
        return inframe
    else:
        for obj in qrobj:
            url = obj.data.decode('utf-8')
            pts = obj.polygon
            pts = np.array([pts],np.int32)
            pts = pts.reshape((4,1,2))
            cv2.polylines(inframe,[pts],True,(255,0,255),2)
            cv2.putText(inframe,url,(50,50),cv2.FONT_HERHEY_PLAIN,1.5,(255,0,0),1.5)
            return inframe
    


while True:

    dic = {" Pink Cuboid" : 2 , "Orange Cone" : 1, "Blue Cylinder" : 3}
    print(dic["Pink Cuboid"])
    frame = cap.read()
    qr = get_qr_code(frame)
    nframe = draw_polygon(frame,qr)
    cv2.imshow("prachit",nframe)
    k = cv2.waitKey(0)
    if k == ord("z"):
        cv2.destroyAllWindows()
    
