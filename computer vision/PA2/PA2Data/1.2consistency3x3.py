# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:52:47 2018

@author: Keerthi
"""
import numpy as np
import cv2

adisp1 = cv2.imread('disp1.png',0)
adisp5 = cv2.imread('disp5.png',0)
disp1 = cv2.imread('disp1_3.jpg',0)
disp5 = cv2.imread('disp5_3.jpg',0)

N=disp1.shape[0]
M=disp1.shape[1]

P=disp5.shape[0]
Q=disp5.shape[1]

consistency1_l = np.zeros((N,M))
consistency1_r = np.zeros((N,M))
consistency5 = np.zeros((P,Q))

for i in range(0,N):
    for j in range(0,M):
        
        left_pixel=int(disp5[i,j])
        temp = j-left_pixel
        if (temp>=1):
            right_pixel=int(disp1[i,temp])
            
            if(left_pixel==right_pixel):
                consistency1_l[i,j]=left_pixel
            else:
                consistency1_l[i,j]=0
                
        right_pixel=int(disp5[i,j])
        temp = j+right_pixel
        if (temp<M):
            left_pixel=int(disp1[i,temp])
            
            if(left_pixel==right_pixel):
                consistency1_r[i,j]=right_pixel
            else:
                consistency1_r[i,j]=0
        
        

cv2.imwrite('consistency3x3_1l.jpg',consistency1_l)                
MSE1 = ((consistency1_l - adisp1) ** 2).mean(axis=None)
print ('MSE for Consistency Check on left image(3x3):',MSE1)

cv2.imwrite('consistency3x3_1r.jpg',consistency1_r)                
MSE = ((consistency1_r - adisp1) ** 2).mean(axis=None)
print ('MSE for Consistency Check on right image(3x3):',MSE)


for i in range(0,P):
    for j in range(0,Q):
        
        left_pixel=int(disp1[i,j])
        temp = j-left_pixel
        if (M>temp>0):
            right_pixel=int(disp5[i,temp])
        else:
            right_pixel=int(disp5[i,j])
            
        if(left_pixel==right_pixel):
            consistency5[i,j]=left_pixel
        else:
            consistency5[i,j]=0
        
        

cv2.imwrite('consistency3x3_5.jpg',consistency5)                
MSE = ((consistency5 - adisp5) ** 2).mean(axis=None)
print ('MSE for Consistency Check on right image(3x3):',MSE)

