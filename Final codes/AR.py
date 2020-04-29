import numpy as np
import cv2
#from matplotlib import pyplot as plt
#import pywavefront
from objloader import *
#import os
import math
#import pymesh
from object_loader_simple import *

MIN_MATCHES = 10

camera_parameters = np.array([[534.34, 0.0, 339.15], [0.0, 534.684, 233.84], [0, 0, 1]])

orb = cv2.xfeatures2d.SIFT_create(2000)

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3)/3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))



FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

flann = cv2.FlannBasedMatcher(index_params,{})      

# trainImage
trainImg = cv2.imread('qrcode.png',0)
trainImg2 = cv2.imread('qrcode2.png',0)
trainImg3 = cv2.imread('qrcode2.png',0)


kpImage, desImage = orb.detectAndCompute(trainImg, None)
kpImage2, desImage2 = orb.detectAndCompute(trainImg2, None)
kpImage3, desImage3 = orb.detectAndCompute(trainImg3, None)

obj = OBJ("cow.obj", swapyz=True)
obj2 = OBJ("deer.obj", swapyz=True)  
obj3 = OBJ("cube.obj", swapyz=True) 

# Video_Aqui
video = cv2.VideoCapture(0) 



_, first_frame = video.read()



while(True):
    
    hasFrame, frame= video.read()    
#    frame = cv2.resize(frame, (640, 480)) 
    BWvideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kpVideo, desVideo = orb.detectAndCompute(BWvideo, None)

    
    # FLANN parameters
    matches = flann.knnMatch(np.asarray(desVideo,np.float32),np.asarray(desImage,np.float32),k=2)
    matches2 = flann.knnMatch(np.asarray(desVideo,np.float32),np.asarray(desImage2,np.float32),k=2)
    matches3 = flann.knnMatch(np.asarray(desVideo,np.float32),np.asarray(desImage3,np.float32),k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
            
    good_matches2 = []
    for m,n in matches2:
        if m.distance < 0.7*n.distance:
            good_matches2.append(m)
            
    good_matches3 = []
    for m,n in matches3:
        if m.distance < 0.7*n.distance:
            good_matches3.append(m)
    
    if len(good_matches) > len(good_matches2):  # and len(good_matches3)):
        if(len(good_matches) > MIN_MATCHES):
            src_pts = []
            dst_pts = []   
            # Assuming matches stores the matches found and 
            # Returned by bf.match(des_model, des_frame)
            # Differenciate between source points and destination points
            src_pts = np.float32([kpImage[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpVideo[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
            src_pts,dst_pts=np.float32((src_pts,dst_pts))
            # Compute Homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                                                      
            # Draw a rectangle that marks the found model in the frame
#            h, w = trainImg.shape
#            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                
            # Project corners into frame
#            dst = cv2.perspectiveTransform(pts, M)
            
            
            
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


    elif len(good_matches2) > len(good_matches): # and len(good_matches3)):       
        if(len(good_matches2) > MIN_MATCHES):
            src_pts = []
            dst_pts = []   
            # Assuming matches stores the matches found and 
            # Returned by bf.match(des_model, des_frame)
            # Differenciate between source points and destination points
            src_pts = np.float32([kpImage2[m.trainIdx].pt for m in good_matches2]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpVideo[m.queryIdx].pt for m in good_matches2]).reshape(-1, 1, 2)
    
            src_pts,dst_pts=np.float32((src_pts,dst_pts))
            # Compute Homography
            M2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                                                      
            # Draw a rectangle that marks the found model in the frame
#            h, w = trainImg2.shape
#            pts2 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                
            # Project corners into frame
#            dst2 = cv2.perspectiveTransform(pts2, M2)  
        
            if M2 is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(camera_parameters, M2) 
                    print('pass')
                    # project cube or model
                    frame = render(frame, obj2, projection, trainImg2, False)
                    #frame = render(frame, model, projection)
                    
                except:
                    pass
        
        
#    elif len(good_matches3) > (len(good_matches) and len(good_matches2)):            
#        if(len(good_matches3) > MIN_MATCHES):
#            src_pts = []
#            dst_pts = []   
#            # Assuming matches stores the matches found and 
#            # Returned by bf.match(des_model, des_frame)
#            # Differenciate between source points and destination points
#            src_pts = np.float32([kpImage3[m.trainIdx].pt for m in good_matches3]).reshape(-1, 1, 2)
#            dst_pts = np.float32([kpVideo[m.queryIdx].pt for m in good_matches3]).reshape(-1, 1, 2)
#    
#            src_pts,dst_pts=np.float32((src_pts,dst_pts))
#            # Compute Homography
#            M3, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#                                                      
#            # Draw a rectangle that marks the found model in the frame
#            #h, w = trainImg3.shape
#            #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#                
#            # Project corners into frame
#            #dst = cv2.perspectiveTransform(pts, M)  
#        
#            if M3 is not None:
#                try:
#                    # obtain 3D projection matrix from homography matrix and camera parameters
#                    projection = projection_matrix(camera_parameters, M3) 
#                    print('pass')
#                    # project cube or model
#                    frame = render(frame, obj3, projection, trainImg3, False)
#                    #frame = render(frame, model, projection)
#                    
#                except:
#                    pass
    else:
        print ("Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCHES))
        matchesMask = None
    
        
    # Display result   
    cv2.imshow('result',np.fliplr(frame))




    if cv2.waitKey(1) & 0xff == 27:
        break

video.release()
cv2.destroyAllWindows()

