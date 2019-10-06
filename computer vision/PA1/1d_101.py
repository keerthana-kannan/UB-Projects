# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 12:42:46 2018

@author: Keerthi
"""

import numpy as np
import cv2
import time

img = cv2.imread('lena_gray.jpg', 0)

img_arr= np.asarray(img)

gx1 = np.random.rand(101,1)
gx2 = np.random.rand(1,101)

K =  gx1.shape[0] 
L = gx2.shape[1]

N = img_arr.shape[0]
M = img_arr.shape[1]

store_gx1 = np.zeros((N,M))
store_gx2 = np.zeros((N,M))

start = time.clock()

img_padx1 = np.pad(img_arr, 50, 'constant')

for i in range(1,N+1):
    for j in range(1,M+1):
        
        gx1_pixel = np.sum(np.multiply(gx1,img_padx1[i-1:i-1+K , j+49:j+50]))
        	   	
        store_gx1[i-1, j-1] = gx1_pixel
  
img_padx2 = np.pad(store_gx1, 50, 'constant')      

for i in range(1,N+1):
    for j in range(1,M+1):
        	
        gx2_pixel = np.sum(np.multiply(gx2,img_padx2[i+49:i+50 , j-1:j-1+L]))
        	
        store_gx2[i-1, j-1] = gx2_pixel


end = time.clock() - start

print("Time taken for 1D convolution:  ", end)
                
cv2.imshow('g', store_gx2)
cv2.waitKey(0)
cv2.destroyAllWindows()