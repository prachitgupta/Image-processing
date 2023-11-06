#!/usr/bin/env python3
import numpy as np
import cv2

img = cv2.imread('vllCR.png',0)
cv2.imshow("prachit",img)
k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('custom_img.png',img)
    cv2.destroyAllWindows()

   