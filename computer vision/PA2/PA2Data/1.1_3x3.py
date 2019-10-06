# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 12:17:01 2018

@author: Keerthi
"""

import numpy as np
import cv2

image1 = cv2.imread('view1.png',0)
image5 = cv2.imread('view5.png',0)
disp1 = cv2.imread('disp1.png',0)
disp5 = cv2.imread('disp5.png',0)

x = image1.shape[0]
y = image1.shape[1]

u = image5.shape[0]
v = image5.shape[1]

pad_image1=cv2.copyMakeBorder(image1,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
pad_image5=cv2.copyMakeBorder(image5,1,1,1,1,cv2.BORDER_CONSTANT,value=0)

disparity_1 = np.zeros((x,y))
disparity_5 = np.zeros((u,v))

m=pad_image1.shape[0]
n=pad_image1.shape[1]
o=pad_image5.shape[1]

for i in range(1,m-1):
    for j in range(1,n-1):
        bestdist= np.inf 
        for k in range(1,o-1):
            temp = np.sum(np.square(pad_image1[i-1:i+2,j-1:j+2]-pad_image5[i-1:i+2,k-1:k+2]))
            if(temp<bestdist):
                bestdist = temp
                disparity_1[i-1][j-1]=abs(j-k)
                if(disparity_1[i-1][j-1]>80):
                    disparity_1[i-1][j-1]=80
                
                
cv2.imwrite('disp1_3.jpg',disparity_1)
MSE = ((disparity_1 - disp1) ** 2).mean(axis=None)
print ('MSE for View1(3x3):',MSE)

m=pad_image5.shape[0]
n=pad_image5.shape[1]
o=pad_image1.shape[1]

for i in range(1,m-1):
    for j in range(1,n-1):
        bestdist= np.inf 
        for k in range(1,o-1):
            temp = np.sum(np.square(pad_image5[i-1:i+2,j-1:j+2]-pad_image1[i-1:i+2,k-1:k+2]))
            if(temp<bestdist):
                bestdist = temp
                disparity_5[i-1][j-1]=abs(j-k)
                if(disparity_5[i-1][j-1]>80):
                    disparity_5[i-1][j-1]=80
                
                
cv2.imwrite('disp5_3.jpg',disparity_5)
MSE = ((disparity_5 - disp5) ** 2).mean(axis=None)
print ('MSE for View5(3x3):',MSE)

consistency1 = np.zeros((disparity_1.shape[0],disparity_1.shape[1]))

for i in range(0,disparity_1.shape[0]):
    for j in range(0,disparity_1.shape[1]):
        
        
        left_pixel=int(disparity_1[i,j])
        if (disparity_1.shape[1]>j-left_pixel>0):
            right_pixel=int(disparity_5[i,j-left_pixel])
       
            if(left_pixel==right_pixel):
                consistency1[i,j]=left_pixel
            else:
                consistency1[i,j]=0

cv2.imwrite('consistency1.jpg',consistency1)      

left=0
for i in range (0,disparity_1.shape[0]):
    for j in range(0,disparity_1.shape[1]):
        if(consistency1[i,j]!=0):
            temp=(pad_image1[i,j]-consistency1[i,j])**2
            left=left+temp


MSE=left/(disparity_1.shape[0]*disparity_1.shape[1])          

print ('MSE for left image after Consistency Check(3x3):',MSE)

consistency2 = np.zeros((disparity_5.shape[0],disparity_5.shape[1]))

for i in range(0,disparity_5.shape[0]):
    for j in range(0,disparity_5.shape[1]):
        
        
        right_pixel=int(disparity_5[i,j])
        if (disparity_5.shape[1]>j+right_pixel>0):
            left_pixel=int(disparity_1[i,j+right_pixel])
    
            if(right_pixel==left_pixel):
                consistency2[i,j]=right_pixel
            else:
                consistency2[i,j]=0

cv2.imwrite('consistency2.jpg',consistency2) 
               
right=0
for i in range (0,disparity_5.shape[0]):
    for j in range(0,disparity_5.shape[1]):
        if(consistency2[i,j]!=0):
            temp=(pad_image5[i,j]-consistency2[i,j])**2
            right=right+temp

MSE=right/(disparity_5.shape[0]*disparity_5.shape[1])

print ('MSE for right image after Consistency Check(3x3):',MSE)
