# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:04:29 2018

@author: Keerthi
"""

import cv2
import numpy as np

left_img1 = cv2.imread('view1.png')  #read it as a grayscale image
right_img1 = cv2.imread('view5.png')

left_img= cv2.cvtColor(left_img1, cv2.COLOR_BGR2GRAY)
right_img= cv2.cvtColor(right_img1, cv2.COLOR_BGR2GRAY)

#Disparity Computation for Left Image
x,y=left_img.shape
u,v=right_img.shape

Disparity_left = np.zeros(left_img.shape, np.uint8)
Disparity_right = np.zeros(right_img.shape, np.uint8)

OcclusionCost = 20 #(You can adjust this, depending on how much threshold you want to give for noise)

numcols = left_img.shape[1]
numrows = left_img.shape[0]
#For Dynamic Programming you have build a cost matrix. Its dimension will be numcols x numcols
for a in range (0,numrows):

    CostMatrix = np.zeros((numcols,numcols))
    DirectionMatrix = np.zeros((numcols,numcols))  #(This is important in Dynamic Programming. You need to know which direction you need traverse)
    
#We first populate the first row and column values of Cost Matrix


    for i in range (0,numcols):
    
        CostMatrix[i,0] = i*OcclusionCost
        CostMatrix[0,i] = i*OcclusionCost


# Now, its time to populate the whole Cost Matrix and DirectionMatrix
    for i in range (1,numcols):
        for j in range(1,numcols):
            abso = np.abs((int(left_img[a,i])-int(right_img[a,j])))
            min1 = CostMatrix[i-1,j-1] + abso
            min2 = CostMatrix[i-1,j] + OcclusionCost
            min3 = CostMatrix[i,j-1] + OcclusionCost
            CostMatrix[i,j] = np.min((min1,min2,min3))
            cmin = CostMatrix[i,j]
             
            if (min1 == cmin):
                DirectionMatrix[i,j] = 1
        
            if (min2 == cmin):
                DirectionMatrix[i,j] = 2
            
            if (min3 == cmin):
                DirectionMatrix[i,j] = 3
            
        p = DirectionMatrix.shape[0]-1
        q = DirectionMatrix.shape[1]-1

    while ((p!=0) and (q!=0)):
        mat = DirectionMatrix[p,q]
        if(mat == 1):
            Disparity_left[a][p] = np.abs(p-q)
            Disparity_right[a][q] = np.abs(p-q)
            p-=1
            q-=1
        elif (mat == 2):
            p-=1
        elif (mat == 3):
            q-=1
        
cv2.imwrite('disparity_left.jpg', Disparity_left)
cv2.imwrite('disparity_right.jpg', Disparity_right)
        
# Use the pseudocode from "A Maximum likelihood Stereo Algorithm" paper given as reference
