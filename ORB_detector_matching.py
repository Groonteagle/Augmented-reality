import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCHES = 10
#
orb = cv2.xfeatures2d.SIFT_create(1500)
#orb = cv2.AKAZEUpright_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,{})      

trainImg = cv2.imread('JEJE.jpg',0) # trainImage
#trainImg2 = cv2.resize(trainImg, (480, 640)) 
kpImage, desImage = orb.detectAndCompute(trainImg, None)

# Video_Aqui
video = cv2.VideoCapture(0) 

while(True):
    
    hasFrame, frame= video.read()    
    BWvideo = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
    kpVideo, desVideo = orb.detectAndCompute(BWvideo, None)
# Initiate SIFT detector
   # orb = cv2.ORB_create()
    
# Find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
    
        # Initiate STAR detector
   
    
# Compute the descriptors with ORB
    

    
# FLANN parameters
    
    
    matches = flann.knnMatch(np.asarray(desVideo,np.float32),np.asarray(desImage,np.float32),k=2)
    
##    # Need to draw only good matches, so create a mask
#    matchesMask = [[0,0] for i in range(len(matches))]
#    
#    if len(matches) > MIN_MATCHES:
#    # ratio test as per Lowe's paper
#        for i,(m,n) in enumerate(matches):
#            if m.distance < 0.7*n.distance:
#                matchesMask[i]=[1,0]
#        
#        draw_params = dict(matchColor = (0,255,0),
#                           singlePointColor = (255,0,0),
#                           matchesMask = matchesMask,
#                           flags = 2)
#        
#        img3 = cv2.drawMatchesKnn(newimg1,kp1,img2,kp2,matches[:MIN_MATCHES],0,flags = 2)
    
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)
            
            
    
    
   #if len(good) > MIN_MATCHES:
        
    if(len(good_matches) > MIN_MATCHES):
        
        src_pts=[]
        dst_pts=[]
    
        
# Assuming matches stores the matches found and 
# Returned by bf.match(des_model, des_frame)
# Differenciate between source points and destination points
        src_pts = np.float32([kpImage[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpVideo[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        #for m in good_matches:
        
#            src_pts.append(kpImage[m.trainIdx].pt)
#            dst_pts.append(kpVideo[m.queryIdx].pt)
    
        src_pts,dst_pts=np.float32((src_pts,dst_pts))
    # Compute Homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                                                  
    # Draw a rectangle that marks the found model in the frame
        h, w = trainImg.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            
    # Project corners into frame
        dst = cv2.perspectiveTransform(pts, M)  
            
    # Connect them with lines
        img2 = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
#        img2 = cv2.polylines(frame, [np.int32(dst)],True,(0,255,0),5)
    else:
        print ("Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCHES))
        matchesMask = None

#    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                       singlePointColor = None,
#                       matchesMask = matchesMask, # draw only inliers
#                       flags = 2)
    
    cv2.imshow('result',frame)
    #img3 = cv2.drawMatches(frame,kpVideo,trainImg2,kpImage,good_matches,0)
    
# Display result
    
    #cv2.imshow('frames',img3)                      
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
video.release()

