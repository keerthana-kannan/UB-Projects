# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 23:39:44 2018

@author: Keerthi
"""

import numpy as np
import cv2
import time

img = cv2.imread('lena_gray.jpg', 0)

img_arr= np.asarray(img)

gx1 = np.array([[1],[2],[1]])
gx2 = np.array([[-1,0,1]])


gy1 = np.array([[-1],[0],[1]])
gy2 = np.array([[1,2,1]])

N = img_arr.shape[0] 
M = img_arr.shape[1] 

store_gx = np.zeros((N,M))
store_gy = np.zeros((N,M))
store_g = np.zeros((N,M))

start = time.clock()

img_pad = np.pad(img_arr, (1,1), 'constant')


for i in range(1,N+1):
    for j in range(1,M+1):
        
        gx_pixel = gx1[0][0] * img_pad[i-1][j] + gx1[1][0] * img_pad[i][j] + gx1[2][0] * img_pad[i+1][j]
        	
        gy_pixel = gy1[0][0] * img_pad[i-1][j] + gy1[1][0] * img_pad[i][j] + gy1[2][0] * img_pad[i+1][j]
        	
        store_gx[i-1, j-1] = gx_pixel
        store_gy[i-1, j-1] = gy_pixel

img_padx = np.pad(store_gx, (1,1), 'constant')
img_pady = np.pad(store_gy, (1,1), 'constant')


for i in range(1,N+1):
    for j in range(1,M+1):
        
        gx_pixel = gx2[0][0] * img_padx[i][j-1] + gx2[0][1] * img_padx[i][j] + gx2[0][2] * img_padx[i][j+1]
        	          
        gy_pixel = gy2[0][0] * img_pady[i][j-1] + gy2[0][1] * img_pady[i][j] + gy2[0][2] * img_pady[i][j+1]
        	
        g_pixel = np.sqrt(gx_pixel * gx_pixel + gy_pixel * gy_pixel)
        
        store_gx[i-1][j-1] = gx_pixel/255
        store_gy[i-1][j-1] = gy_pixel/255
        store_g[i-1][j-1] = g_pixel/255


end = time.clock() - start

print("Time taken for 1D convolution:  ", end)
                
cv2.imshow('gx', store_gx)


cv2.imshow('gy', store_gy)


cv2.imshow('g', store_g)

cv2.waitKey(0)
cv2.destroyAllWindows()
        