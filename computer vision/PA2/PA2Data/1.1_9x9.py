# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:31:14 2018

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

pad_image1=cv2.copyMakeBorder(image1,4,4,4,4,cv2.BORDER_CONSTANT,value=0)
pad_image5=cv2.copyMakeBorder(image5,4,4,4,4,cv2.BORDER_CONSTANT,value=0)

disparity_1 = np.zeros((x,y))
disparity_5 = np.zeros((u,v))

m=pad_image1.shape[0]
n=pad_image1.shape[1]
o=pad_image5.shape[1]

for i in range(4,m-4):
    for j in range(4,n-4):
        bestdist= np.inf 
        for k in range(4,o-4):
            temp = np.sum(np.square(pad_image1[i-4:i+5,j-4:j+5]-pad_image5[i-4:i+5,k-4:k+5]))
            if(temp<bestdist):
                bestdist = temp
                disparity_1[i-4][j-4]=abs(j-k)
                if(disparity_1[i-4][j-4]>80):
                    disparity_1[i-4][j-4]=80
                
                
cv2.imwrite('disp1_9.jpg',disparity_1)
MSE = ((disparity_1 - disp1) ** 2).mean(axis=None)
print ('MSE for View1(9x9):',MSE)

m=pad_image5.shape[0]
n=pad_image5.shape[1]
o=pad_image5.shape[1]

for i in range(4,m-4):
    for j in range(4,n-4):
        bestdist= np.inf 
        for k in range(4,o-4):
            temp = np.sum(np.square(pad_image5[i-4:i+5,j-4:j+5]-pad_image1[i-4:i+5,k-4:k+5]))
            if(temp<bestdist):
                bestdist = temp
                disparity_5[i-4][j-4]=abs(j-k)
                if(disparity_5[i-4][j-4]>80):
                    disparity_5[i-4][j-4]=80
                
                
cv2.imwrite('disp5_9.jpg',disparity_5)
MSE = ((disparity_5 - disp5) ** 2).mean(axis=None)
print ('MSE for View5(9x9):',MSE)



consistency1=np.zeros((disparity_1.shape[0],disparity_1.shape[1]))
consistency2=np.zeros((disparity_5.shape[0],disparity_5.shape[1]))

for i in range (4,disparity_1.shape[0]-4):
    for j in range (4, disparity_1.shape[1]-4):
        left_pixel=int(disparity_1[i,j])
        if j-left_pixel>0:
            right_pixel=int(disparity_5[i,j-left_pixel])
      
            if(left_pixel==right_pixel):
                consistency1[i,j]=left_pixel
            else:
                consistency1[i,j]=0

cv2.imwrite('consistency1_9.jpg',consistency1)

left=0
for i in range (0,disparity_1.shape[0]):
    for j in range(0,disparity_1.shape[0]):
        if(consistency1[i,j]!=0):
            temp=(pad_image1[i,j]-consistency1[i,j])**2
            left=left+temp


MSE=left/(disparity_1.shape[0]*disparity_1.shape[1])
print ("MSE of Left Image after Consistency check(9x9):" , MSE)


for i in range (4,disparity_1.shape[0]-4):
    for j in range (4, disparity_1.shape[1]-4):
        right_pixel=int(disparity_5[i,j])
        if j+right_pixel<disparity_1.shape[1]-4:
            
            left_pixel=int(disparity_1[i,j+right_pixel])
 
                
            if(right_pixel==left_pixel):
                consistency2[i,j]=right_pixel
            else:
                consistency2[i,j]=0


cv2.imwrite('consistency2_9.jpg',consistency2)

right=0
for i in range (0,disparity_1.shape[0]):
    for j in range(0,disparity_1.shape[1]):
        if(consistency2[i,j]!=0):
            temp=(pad_image5[i,j]-consistency2[i,j])**2
            right=right+temp

MSE=right/(disparity_1.shape[0]*disparity_1.shape[1])
print ("MSE of Right Image after Consistency check(9x9):" , MSE)
