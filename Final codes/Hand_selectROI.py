# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:29:19 2019

@author: lilis
"""

import numpy as np
import cv2
import math
import imutils

# LOAD HAND CASCADE----------------------------------------------------------------
hand_cascade = cv2.CascadeClassifier('Hand_haar_cascade.xml')

# VIDEO CAPTURE--------------------------------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # SKIN DETECTION---------------------------------------------------------------
    # Convert frame to HSV space   
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define upper and lower boundaries of the HSV pixel
    # intensities to be considered as 'skin'
    hsv_lower = np.array([0, 0, 0], dtype = "uint8")
    hsv_upper = np.array([255, 80, 255], dtype = "uint8")
    mask_skin_hsv = cv2.inRange(frame_hsv, hsv_lower, hsv_upper)
    
    # Construction of the mask for the skin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_skin_hsv = cv2.erode(mask_skin_hsv, kernel, iterations=2)    
    
    # Filter the frame to get the skin
    skin_hsv = cv2.bitwise_and(frame, frame, mask=mask_skin_hsv)
    
    # Define upper and lower boundaries of the RGB pixel
    # intensities to be considered as 'skin'
    rgb_lower = np.array([50, 0, 0], dtype = "uint8")
    rgb_upper = np.array([255, 255, 255], dtype = "uint8")
    mask_skin_rgb = cv2.inRange(skin_hsv, rgb_lower, rgb_upper)

    # Filter the previous skin found with HSV to get the  final skin
    skin_rgb = cv2.bitwise_and(skin_hsv, skin_hsv, mask=mask_skin_rgb)



    #cv2.imshow('frame', np.fliplr(frame))
    #cv2.imshow('frameHSV', np.fliplr(frame_hsv))
    cv2.imshow('frame skinHSV', np.fliplr(skin_hsv))
    #cv2.imshow('frame skinRGB', np.fliplr(skin_rgb))
    
    
    
    
    
    
    skin_rgb = cv2.cvtColor(skin_rgb, cv2.COLOR_BGR2GRAY)
    skin_rgb = imutils.auto_canny(skin_rgb)
    
    image, contours, hierarchy = cv2.findContours(skin_rgb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x,y,w,h = cv2.boundingRect(cnt)
            frame3 = cv2.drawContours(frame, cnt, -1, (0,255,0), 2)
            frame3 = cv2.rectangle(frame3,(x,y),(x+w,y+h),(0,0,255),1)
    
    
    
    
    # Apply Haar cascade to detect hands
    #skin_rgb = cv2.cvtColor(skin_rgb, cv2.COLOR_BGR2GRAY)
    skin_rgb_eq = cv2.equalizeHist(skin_rgb)
    hand = hand_cascade.detectMultiScale(skin_rgb_eq, 1.1, 5)
    
    frame2 = frame.copy()
    for (x,y,w,h) in hand: # MARKING THE DETECTED ROI
    	cv2.rectangle(frame2,(x-30,y-30),(x+w+30,y+h+30), (0,255,0), 1) 
    
    #cv2.imshow('frame skin gray', np.fliplr(skin_rgb))

    #cv2.imshow('frame hand detected', np.fliplr(frame3))
    

    
    
    
    
    
    
#    # BGR to HSV
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    
#    # Blur images to get smoothen edges
#    blur = cv2.GaussianBlur(hsv,(5,5),0)
#    
#    # Gray scale image
#    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#    
#    # Binary image
#    retval2,thresh1 = cv2.threshold(gray,65,255,cv2.THRESH_BINARY)
#    
#    cv2.imshow('frame', np.fliplr(frame))
#    cv2.imshow('hsv', np.fliplr(hsv))
#    cv2.imshow('blurry', np.fliplr(blur))
#    cv2.imshow('gray', np.fliplr(gray))
#    cv2.imshow('binary', np.fliplr(thresh1))
    
    
     #skeletonize the image
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    skeleton = imutils.skeletonize(gray, size=(5, 5))
#    
#    cv2.imshow('skeleton', np.fliplr(skeleton))
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()





















