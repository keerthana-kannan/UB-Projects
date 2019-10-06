# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:23:39 2018

@author: Keerthi
"""
import numpy as np
import cv2


disp1=cv2.imread('disp1.png',0);
disp5=cv2.imread('disp5.png',0);
disparity_map1=cv2.imread('disp1_3.jpg',0);
disparity_map5=cv2.imread('disp1_5.jpg',0);
view1 = cv2.imread('view1.png',0);
view5 = cv2.imread('view5.png',0);

view1_with_border = cv2.copyMakeBorder(view1, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=0)
view5_with_border = cv2.copyMakeBorder(view5, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=0)
disp1_border=cv2.copyMakeBorder(disp1, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=0)
disp5_border=cv2.copyMakeBorder(disp5, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=0)

x,y=view1_with_border.shape
v,w=view5_with_border.shape

consistency_left=np.zeros((x,y))
consistency_right=np.zeros((v,w))




#LEFT

for i in range (0,x):
    for j in range (0, y):
        pixel_value_l=disparity_map1[i,j]
        if y>j-pixel_value_l>0:
            pixel_value_r=disparity_map5[i,j-pixel_value_l]
            
        if(pixel_value_l==pixel_value_r):
            consistency_left[i,j]=pixel_value_l
        else:
            consistency_left[i,j]=0

cv2.imshow("consistency_3x3_75L",consistency_left)
cv2.imwrite('consistency3X3_75_Left.png',consistency_left)


Sum_left=0
for i in range (0,x):
    for j in range(0,y):
        if(consistency_left[i,j]!=0):
            temp=(disp1_border[i,j]-consistency_left[i,j])**2
            Sum_left=Sum_left+temp


mse_c_left=Sum_left/(x*y)
print ("MSE with respect to Leftt Image when the block is 3X3 after Consistency check" , mse_c_left)
