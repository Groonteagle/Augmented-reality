# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 18:57:13 2019

@author: lilis
"""


import numpy as np
import cv2
from AR_functions import *
from object_loader_simple import *


# Define initial parameters
MIN_MATCHES = 5
camera_parameters = np.array([[534.34, 0.0, 339.15], [0.0, 534.684, 233.84], [0, 0, 1]])

#LOADING FACE HAAR CASCADE
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# VIDEO CAPTURE
cap = cv2.VideoCapture(0)

# Create a sift
sift = cv2.xfeatures2d.SIFT_create(2000)

# Get the first frame of the video
_, first_frame = cap.read()





# Get gray image
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)




faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(first_frame,(x+10,y+10),(x+w-10,y+h-10),(255,0,0),2)
    roi_color = first_frame[y+10:y+h-10, x+10:x+w-10]

# Convert ROI into binary
roi_color_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
roi_color_gray = cv2.threshold(roi_color_gray, 50, 255, cv2.THRESH_BINARY)
roi_color_gray = roi_color_gray[1]

# Compute the sift
kp, des = sift.detectAndCompute(roi_color_gray,None)
cv2.drawKeypoints(roi_color,kp,roi_color,(0,0,255),2)

cv2.imshow('img',roi_color)
cv2.waitKey(0)
cv2.destroyAllWindows()




#%%


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

flann = cv2.FlannBasedMatcher(index_params,{})      

trainImg = roi_color
trainImg = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)

roi_color2 = []

# Read 3D image
obj = OBJ("cow.obj", swapyz=True)

# Create a sift
sift2 = cv2.xfeatures2d.SIFT_create(2000)

while True:
    _, frame = cap.read()   # Open the camera
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)            
    
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for rect in faces:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, (x+10, y+10), (x+w-10, y+h-10), (0, 255, 0), 2)
        roi_color2 = frame[y+10:y+h-10, x+10:x+w-10]
    
    
    
    
    
    
    # Convert ROI into binary
    roi_color_gray2 = cv2.cvtColor(roi_color2, cv2.COLOR_BGR2GRAY)
    roi_color_gray2 = cv2.threshold(roi_color_gray2, 50, 255, cv2.THRESH_BINARY)
    roi_color_gray2 = roi_color_gray2[1]
        
    # Compute the sift
    kp2, des2 = sift2.detectAndCompute(roi_color_gray2,None)
    
    
    
    
    
    
    # FLANN parameters
    matches = flann.knnMatch(np.asarray(des2,np.float32),np.asarray(des,np.float32),k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
        
    if(len(good_matches) > MIN_MATCHES):
        
        src_pts = []
        dst_pts = []   
# Assuming matches stores the matches found and 
# Returned by bf.match(des_model, des_frame)
# Differenciate between source points and destination points
        src_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts,dst_pts=np.float32((src_pts,dst_pts))
        
    # Compute Homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if len(trainImg.shape) > 0:                                      
            # Draw a rectangle that marks the found model in the frame
            h, w = trainImg.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            
            # Project corners into frame
            dst = cv2.perspectiveTransform(pts, M)  
            
    # Connect them with lines
        #cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
        
        if M is not None:
            try:
                # obtain 3D projection matrix from homography matrix and camera parameters
                projection = projection_matrix(camera_parameters, M) 
                print('pass')
                # project cube or model
                frame = render(frame, obj, projection, trainImg, False)
                #frame = render(frame, model, projection)
                
            except:
                pass
            
    else:
        print ("Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCHES))
        matchesMask = None
    
    
        
    cv2.imshow("Frame", np.fliplr(frame))
    if cv2.waitKey(1) & 0xFF == 27:   # press Q to quit the window
        break
    
cap.release()
cv2.destroyAllWindows()








































