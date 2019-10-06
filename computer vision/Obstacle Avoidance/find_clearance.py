# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 08:50:13 2019

@author: Keerthi
"""


import numpy as np
import cv2
import sys

iii = open(sys.argv[1], "r")
image = np.loadtxt(iii)
iii.close()

w = image.shape[0] #176
h =image.shape[1] #132
a = np.zeros((w,h), dtype=np.uint8)
b = np.zeros((w,h), dtype=np.uint8)

#automatic canny edge detection (zero parameter)
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

#reduce the noise in the image
def denoise(image,lower,upper):
    for i in range(w):
        for j in range(h):
            if (image[i,j] < lower):
                a[i,j] = 0
            elif(image[i,j] > upper):
                a[i,j] = 255
            else:
                a[i,j]=image[i,j]
    return a

#preprocessing the image
a = denoise(image,1,4)
blurred = cv2.GaussianBlur(a, (5,5), 0)
edged = auto_canny(blurred)
edged = cv2.dilate(edged, None, iterations=2)
edged = cv2.erode(edged, None, iterations=2)

#cropping the area of interest
b[25:100,55:120] = edged[25:100,55:120] #[Y:X]

# find contours in the edge map
(_, cnts, _) = cv2.findContours(b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1] #sorting the contours
cnts_polygon = [None]*len(cnts) #apply approximation to polygons with accuracy +-3
boundRect = [None]*len(cnts) #bounding rectangle for every polygon
for (i, c) in enumerate(cnts):
    cnts_polygon[i] = cv2.approxPolyDP(c, 3, True) #3 = max dist between original curve and approx ,closed curve-true
    boundRect[i] = cv2.boundingRect(cnts_polygon[i])
    
#distance from shelf
dist1 = boundRect[0][0]-55
# distance from wall
dist2 = 120-(boundRect[0][0]+boundRect[0][2])

#assuming 1pixel = 1.5m/60 X = 0.025
if dist1 > dist2:
    print ("left" , (dist1*0.025))
    
else:
    print ("right", (dist2*0.025))




            

            

    
