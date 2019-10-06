# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:17:00 2018

@author: Keerthi
"""

import numpy as np
import cv2

image = cv2.imread('Butterfly.jpg');

N=image.shape[0]
M=image.shape[1]
val=image.shape[2]

feature = np.zeros((N*M,5))
seg_image = np.zeros((N,M,val))

h = 60
iters= 30
counter = 0
mean = np.array([])

for i in range (0,N):
    for j in range(0,M):
        feature[counter][0]=image[i][j][0]
        feature[counter][1]=image[i][j][1]
        feature[counter][2]=image[i][j][2]
        feature[counter][3]=i
        feature[counter][4]=j
    
        counter=counter+1
        
while(feature.size!=0):
    if(mean.size == 0):
        mean = feature[0]
        
    cluster = []
    new_mean = 0
    counter=0
    
    for i in range(0,feature.shape[0]):
        
        e_distance = np.sqrt(np.sum((mean-feature[i])**2))
        
        if(e_distance<h):
            cluster.append(i)
            new_mean+= feature[i]
            counter = counter+1
            
    new_mean/= counter
    
    dist =np.sqrt(np.sum((new_mean-mean)**2))
    
    if(dist<=iters):
        for a in range(0,len(cluster)):
          
            seg_image[int(feature[cluster[a]][3]),int(feature[cluster[a]][4])] = np.array([new_mean[0],new_mean[1],new_mean[2]])   
            
        feature =np.delete(feature,cluster,0)
        mean = np.array([])
    else:
        mean = new_mean
                
cv2.imwrite('MeanShift3.jpg',seg_image) 
    
    
    
    
    
    
    
    
    
    