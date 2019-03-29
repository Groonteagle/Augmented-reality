import numpy as np
import sys
import cv2
from matplotlib import pyplot as plt

#img = cv2.imread('amslergrid_test.jpg',)

cam = cv2.VideoCapture(0)


while(True):
    
    
    hasFrame, img = cam.read()
    
    # Initiate STAR detector
    
    orb = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, img, color=(0,0,255), flags=0)
    imS = cv2.resize(img2, (1024, 1024)) 
    cv2.imshow('keypoints',imS)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
cam.release()


