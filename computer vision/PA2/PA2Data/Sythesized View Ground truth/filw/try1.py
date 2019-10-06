# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:41:56 2018

@author: Keerthi
"""

import cv2,numpy as np

img1 = cv2.imread('view1.png',0)
img5 = cv2.imread('view5.png',0)
disp1 = cv2.imread('disp1.png',0)
disp5 = cv2.imread('disp5.png',0)

disp1gen3 = np.zeros((img1.shape[0],img1.shape[1]))
disp5gen3 = np.zeros((img5.shape[0],img5.shape[1]))
padded3img1 = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
padded3img5 = cv2.copyMakeBorder(img5,1,1,1,1,cv2.BORDER_CONSTANT,value=0)

for i in range(1,padded3img1.shape[0]-1):
    for j in range(1,padded3img1.shape[1]-1):
        min = np.inf
        if(j<75):
            start = 1
            end = padded3img5.shape[1]-1
        else:
            end = j+1
            start = j-74
        for k in range(start,end):
            temp = np.sum(np.square(padded3img1[i-1:i+2,j-1:j+2]-padded3img5[i-1:i+2,k-1:k+2]))
            if(temp<min):
                min = temp
                disp1gen3[i-1][j-1]=abs(j-k)
        if(disp1gen3[i-1][j-1]>74):
            disp1gen3[i-1][j-1] = 74
                
cv2.imwrite('disp1gen3.jpg',disp1gen3)
view13MSE = ((disp1gen3 - disp1) ** 2).mean(axis=None)

for i in range(1,padded3img5.shape[0]-1):
    for j in range(1,padded3img5.shape[1]-1):
        min = np.inf
        if((padded3img1.shape[1]-j-1)<76):
            start = 1
            end = padded3img1.shape[1]-1
        else:
            end = j+76
            start = j
        for k in range(start,end):
            temp = np.sum(np.square(padded3img5[i-1:i+2,j-1:j+2]-padded3img1[i-1:i+2,k-1:k+2]))
            if(temp<min):
                min = temp
                disp5gen3[i-1][j-1]=abs(j-k)
        if(disp5gen3[i-1][j-1]>74):
            disp5gen3[i-1][j-1] = 74
                
cv2.imwrite('disp5gen3.jpg',disp5gen3)
view53MSE = ((disp5gen3 - disp5) ** 2).mean(axis=None)

disp1gen9 = np.zeros((img1.shape[0],img1.shape[1]))
disp5gen9 = np.zeros((img5.shape[0],img5.shape[1]))
padded9img1 = cv2.copyMakeBorder(img1,4,4,4,4,cv2.BORDER_CONSTANT,value=0)
padded9img5 = cv2.copyMakeBorder(img5,4,4,4,4,cv2.BORDER_CONSTANT,value=0)

for i in range(4,padded9img1.shape[0]-4):
    for j in range(4,padded9img1.shape[1]-4):
        min = np.inf
        if(j<79):
            start = 4
            end = padded9img5.shape[1]-4
        else:
            end = j+1
            start = j-74
        for k in range(start,end):
            temp = np.sum(np.square(padded9img1[i-4:i+5,j-4:j+5]-padded9img5[i-4:i+5,k-4:k+5]))
            if(temp<min):
                min = temp
                disp1gen9[i-4][j-4]=abs(j-k)
        if(disp1gen9[i-4][j-4]>74):
            disp1gen9[i-4][j-4] = 74
                
cv2.imwrite('disp1gen9.jpg',disp1gen9)
view19MSE = ((disp1gen9 - disp1) ** 2).mean(axis=None)

for i in range(4,padded9img5.shape[0]-4):
    for j in range(4,padded9img5.shape[1]-4):
        min = np.inf
        if((padded9img1.shape[1]-j-4)<79):
            start = 4
            end = padded9img1.shape[1]-4
        else:
            end = j+76
            start = j
        for k in range(start,end):
            temp = np.sum(np.square(padded9img5[i-4:i+5,j-4:j+5]-padded9img1[i-4:i+5,k-4:k+5]))
            if(temp<min):
                min = temp
                disp5gen9[i-4][j-4]=abs(j-k)
        if(disp5gen9[i-4][j-4]>79):
            disp5gen9[i-4][j-4] = 74
                
cv2.imwrite('disp5gen9.jpg',disp5gen9)
view59MSE = ((disp5gen9 - disp5) ** 2).mean(axis=None)






cc3 = np.zeros((disp5gen3.shape[0],disp5gen3.shape[1]))
cc31 = np.zeros((disp1gen3.shape[0],disp1gen3.shape[1]))

for i in range(0,disp1gen3.shape[0]):
    for j in range(0,disp1gen3.shape[1]):
        
        
        pixel_value_l=int(disp1gen3[i,j])
        if (disp1gen3.shape[1]>j-pixel_value_l>0):
            pixel_value_r=int(disp5gen3[i,j-pixel_value_l])
        else:
            pixel_value_r=int(disp5gen3[i,j])
            
        if(pixel_value_l==pixel_value_r):
            cc31[i,j]=pixel_value_l
        else:
            cc31[i,j]=0
        
        
        
        
        """   
        
        temp=int(j+disp5gen3[i][j])
        if(temp<463):
            if(int(disp5gen3[i][j]) == int(disp1gen3[i][temp])):
                cc3[i][j] = disp5gen3[i][j]
                
        """

cv2.imwrite('cc3.jpg',cc31)                
cc3MSE = ((cc31 - disp5) ** 2).mean(axis=None)

cc9 = np.zeros((disp5gen9.shape[0],disp5gen9.shape[1]))

"""
for i in range(0,disp5gen9.shape[0]):
    for j in range(0,disp5gen9.shape[1]):
        
        temp=j+disp5gen9[i][j]
        if(temp<463):
            if(disp5gen9[i][j] == disp1gen9[i][temp]):
                cc9[i][j] = disp5gen9[i][j]
"""

for i in range(0,disp5gen9.shape[0]):
    for j in range(0,disp5gen9.shape[1]):
        pixel_value_l=int(disp1gen9[i,j])
        if j-pixel_value_l>0:
            pixel_value_r=int(disp5gen9[i,j-pixel_value_l])
        else:
            pixel_value_r=int(disp5gen9[i,j])
            
        if(pixel_value_l==pixel_value_r):
            cc9[i,j]=pixel_value_l
        else:
            cc9[i,j]=0

cv2.imwrite('cc9.jpg',cc9)                
cc9MSE = ((cc9 - disp5) ** 2).mean(axis=None)

print ('Mean Square Error for 3X3 View1:',view13MSE)
print ('Mean Square Error for 3X3 View5:',view53MSE)
print ('Mean Square Error for 3X3 Consistency Check:',cc3MSE)
print ('Mean Square Error for 9X9 View1:',view19MSE)
print ('Mean Square Error for 9X9 View5:',view59MSE)
print ('Mean Square Error for 9X9 Consistency Check:',cc9MSE)