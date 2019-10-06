# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:44:59 2018

@author: Keerthi
"""

import cv2
import numpy as np

image1 = cv2.imread('view1.png')
image5 = cv2.imread('view5.png')


disp1 = cv2.imread('disp1.png',0)
disp5 = cv2.imread('disp5.png',0)

N=image1.shape[0]
M=image1.shape[1]
val=image1.shape[2]

view3 = np.zeros((N,M,val))
displace = np.full((N,M),-1)

disp1 = disp1//2
disp5 = disp5//2

x=disp1.shape[0]
y=disp1.shape[1]

for i in range(0,x):
    for j in range(0,y):
        temp = j-disp1[i][j]
        if(temp>=0):
            if(displace[i][temp]<disp1[i][j]):
                displace[i][temp]=disp1[i][j]
                view3[i][temp]=image1[i][j]
                if(view3[i][temp] == 0):
                    temp = j+disp5[i][j]
                    view3[i][temp]=image5[i][j]
                    

u=disp5.shape[0]
v=disp5.shape[1]
                
for i in range(0,u):
    for j in range(0,v):
        temp = j+disp5[i][j]
        if(temp<v):
            if(displace[i][temp]<disp5[i][j]):
                displace[i][temp]=disp5[i][j]
                view3[i][temp]=image5[i][j]
                

cv2.imwrite('viewsynthesis.jpg',view3) 
