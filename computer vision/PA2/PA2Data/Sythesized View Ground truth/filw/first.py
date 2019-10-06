# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:29:41 2018

@author: Keerthi
"""

import cv2
import numpy as np

img1 = cv2.imread('view1.png',0)
img2 = cv2.imread('view5.png',0)
dp1 = cv2.imread('disp1.png',0)
dp2 = cv2.imread('disp5.png',0)
img1=cv2.copyMakeBorder(img1,4,4,4,4,cv2.BORDER_CONSTANT,value=0)
img2=cv2.copyMakeBorder(img2,4,4,4,4,cv2.BORDER_CONSTANT,value=0)
dp1=cv2.copyMakeBorder(dp1,4,4,4,4,cv2.BORDER_CONSTANT,value=0)
dp2=cv2.copyMakeBorder(dp2,4,4,4,4,cv2.BORDER_CONSTANT,value=0)

displft3=np.zeros((378,471))
disprgt3=np.zeros((378,471))
displft9=np.zeros((378,471))
disprgt9=np.zeros((378,471))
bpl=np.zeros((378,471))
bpr=np.zeros((378,471))
for i in range (4,374):
        print (i)
        for j in range(4,467):
            min_ssdl3=99999
            min_ssdl9=99999
            min_ssdr3=99999
            min_ssdr9=99999
            matchl3=0
            matchl9=0
            matchr3=0
            matchr9=0
            k=j
            while(j-k<75)and (k!=0):
                ssdl3=0
                for l in range(-4,5):
                    for m in range (-4,5):
                        diffl3=img1[i+l,j+m] - img2[i+l,k+m]  #9x9 block #view1-view5
                        dfsql3=diffl3 * diffl3
                        ssdl3=ssdl3+dfsql3                        
                if(ssdl3<min_ssdl3):
                    min_ssdl3=ssdl3
                    matchl3=k
                k=k-1   
            displft3[i,j]=abs(j-matchl3)
           
            
            for k in range(j,j+75):
                if(k<467):
                    ssdr3=0
                    for l in range(-4,5):
                        for m in range (-4,5):                        
                            diffr3=img2[i+l,j+m] - img1[i+l,k+m]  #9x9 block #view5-view1
                            dfsqr3=diffr3 * diffr3
                            ssdr3=ssdr3+dfsqr3           
                    if(ssdr3<min_ssdr3):
                        min_ssdr3=ssdr3
                        matchr3=k
            disprgt3[i,j]=matchr3-j

            
cv2.namedWindow('disparity_lft:9 x 9 block', cv2.WINDOW_NORMAL)
cv2.imshow('disparity_lft:9 x 9 block',displft3/displft3.max())
cv2.namedWindow('disparity_rgt:9 x 9 block', cv2.WINDOW_NORMAL)
cv2.imshow('disparity_rgt:9 x 9 block',disprgt3/disprgt3.max())


#back projection
for a in range(4,374):
    for b in range(4,467):
        d1l=displft3[a,b]
        if(b-d1l>=4):
            d1r=disprgt3[a,b-d1l]
            if(d1l!=d1r):
                bpl[a,b]=0
            else:
                bpl[a,b]=displft3[a,b]        
        d2r=disprgt3[a,b]
        if(b+d2r<468):
            d2l=displft3[a,b+d2r]
            if(d2l!=d2r):
                bpr[a,b]=0
            else:
                bpr[a,b]=disprgt3[a,b]
                

cv2.namedWindow('backproj_left', cv2.WINDOW_NORMAL)
cv2.imshow('backproj_left',bpl/bpl.max())
cv2.namedWindow('backproj_rgt', cv2.WINDOW_NORMAL)
cv2.imshow('backproj_rgt',bpr/bpr.max())

#MSE
smdl=0
smdr=0
smbpl=0
smbpr=0
for x in range(1,371):
    for y in range(1,464):
        dfdl=dp1[x,y] - displft3[x,y]    
        sqdl=dfdl * dfdl
        smdl=smdl+sqdl
        dfdr=dp2[x,y] - disprgt3[x,y]    
        sqdr=dfdr * dfdr
        smdr=smdr+sqdr
        if(bpl[x,y]!=0):
            dfbpl=dp1[x,y] - bpl[x,y]    
            sqbpl=dfbpl * dfbpl
            smbpl=smbpl+sqbpl
        if(bpr[x,y]!=0):
            dfbpr=dp2[x,y] - bpr[x,y]    
            sqbpr=dfbpr * dfbpr
            smbpr=smbpr+sqbpr
            
        
mse_disp_lft=smdl/(370*463)
mse_disp_rgt=smdr/(370*463)
mse_bp_lft=smbpl/(370*463)
mse_bp_rgt=smbpr/(370*463)

print ('mse_disp_lft:')
print (mse_disp_lft)
print ('mse_disp_rgt:')
print (mse_disp_rgt)
print ('mse_bp_lft:')
print (mse_bp_lft)
print ('mse_bp_rgt:')
print (mse_bp_rgt)