# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 10:52:18 2018

@author: Keerthi
"""

import numpy as np
import cv2
import time

img = cv2.imread('lena_gray.jpg', 0)

img_arr= np.asarray(img)

gx1 = np.random.rand(101,1)
gx2 = np.random.rand(1,101)
gx = np.outer(gx1, gx2)


K =  gx.shape[0] 
L = gx.shape[1]

N = img_arr.shape[0]
M = img_arr.shape[1]

store_gx = np.zeros((N,M))


#convolution begins

start = time.clock()


img_pad = np.pad(img_arr,50, 'constant')


for i in range(1,N+1):
    for j in range(1,M+1):
        
        gx_pixel = np.sum(np.multiply(gx,img_pad[i-1:i-1+K , j-1:j-1+L]))
        
        store_gx[i-1,j-1] = gx_pixel
       
        
end = time.clock() - start

print("Time taken for 2D convolution: ", end)

cv2.imshow('g',store_gx)
cv2.waitKey(0)
cv2.destroyAllWindows()

