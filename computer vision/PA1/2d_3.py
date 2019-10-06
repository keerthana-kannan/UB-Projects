# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 20:52:12 2018

@author: Keerthi
"""

import numpy as np
import cv2
import time

# Load an color image in grayscale
img = cv2.imread('lena_gray.jpg',0)

img_arr= np.asarray(img)


gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])


N = img_arr.shape[0]
M = img_arr.shape[1]



store_gx = np.zeros((N,M))
store_gy = np.zeros((N,M))
store_g = np.zeros((N,M))

#convolution begins

start = time.clock()


img_pad = np.pad(img_arr, (1,1), 'constant')

X = img_pad.shape[0] 
Y = img_pad.shape[1]


for i in range(1, X-1):
    for j in range(1,Y-1):
        
        gx_pixel = (gx[0][0]*img_pad[i-1][j-1]) + (gx[0][1]*img_pad[i-1][j])+ (gx[0][2]*img_pad[i-1][j+1])+ (gx[1][0]*img_pad[i][j-1])+ (gx[1][1]*img_pad[i][j])+ (gx[1][2]*img_pad[i][j+1])+ (gx[2][0]*img_pad[i+1][j-1])+ (gx[2][1]*img_pad[i+1][j])+ (gx[2][2]*img_pad[i+1][j+1])
        gy_pixel = (gy[0][0]*img_pad[i-1][j-1]) + (gy[0][1]*img_pad[i-1][j])+ (gy[0][2]*img_pad[i-1][j+1])+ (gy[1][0]*img_pad[i][j-1])+ (gy[1][1]*img_pad[i][j])+ (gy[1][2]*img_pad[i][j+1])+ (gy[2][0]*img_pad[i+1][j-1])+ (gy[2][1]*img_pad[i+1][j])+ (gy[2][2]*img_pad[i+1][j+1])
        
        
        g_pixel = np.sqrt(gx_pixel * gx_pixel + gy_pixel * gy_pixel)
        
        store_gx[i-1][j-1] = gx_pixel/255
        store_gy[i-1][j-1] = gy_pixel/255
        store_g[i-1][j-1] = g_pixel/255
        
end = time.clock() - start

print("Time taken for 2D convolution: ", end)
        
cv2.imshow('gx', store_gx)


cv2.imshow('gy', store_gy)


cv2.imshow('g', store_g)

cv2.waitKey(0)
cv2.destroyAllWindows()
