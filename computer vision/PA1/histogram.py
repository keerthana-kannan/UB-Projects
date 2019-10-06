# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 10:30:10 2018

@author: Keerthi
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('dog.jpg')

image = cv2.imread('dog.jpg', 0)

N = image.shape[0]
M = image.shape[1]

H = np.zeros(256)
H_c = np.zeros(256)
T = np.zeros(256)
eqH = np.zeros(256)
eqH_c = np.zeros(256)

for p in range(0, N):
	for q in range(0, M):
		H[image[p][q]] += 1  


H_c[0] = H[0]
for p in range(1, 256):
	H_c[p] = H_c[p-1] + H[p]

G = 256.0
look_up = (G - 1)/(N * M)
for p in range(1, 256):
	T[p] = round(look_up * H_c[p])


for p in range(0, N):
	for q in range(0, M):
		image[p][q] = T[image[p][q]]
        
P = image.shape[0]
Q = image.shape[1]

for p in range(0, P):
	for q in range(0, Q):
		eqH[image[p][q]] += 1  


x = np.arange(256)


plt.figure(1) 
plt.bar(x, H, align='center')
plt.xticks(np.arange(0, 256, 20.0))
plt.xlabel('Intensity Value')
plt.ylabel('Number of pixels')
plt.title('Gray Image Histogram')


plt.figure(2) 
plt.bar(x, H_c, align='center')
plt.xticks(np.arange(0, 256, 20.0))
plt.xlabel('Intensity Value')
plt.ylabel('Number of pixels')
plt.title('Cumulative Historgram')


plt.figure(3) 
plt.plot(T)
plt.xlabel('Original intensity value')
plt.ylabel('New intensity value')
plt.title('Transformation Function')


plt.figure(4) 
plt.bar(x, eqH, align='center')
plt.xticks(np.arange(0, 256, 20.0))
plt.xlabel('Intensity Value')
plt.ylabel('Number of pixels')
plt.title('Equalized Historgram')
plt.show()

cv2.imshow("Original Image",img)


cv2.imshow("Contrast Enhanced Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()


