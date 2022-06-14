import cv2
import numpy as np
from matplotlib import pyplot as plt
color = ['b','g','r']

img = cv2.imread('E:/Nasa Images/nasa62954.jpg',0)

mask = np.zeros(img.shape[:2],np.uint8)
x,y = img.shape[:2]
mask[x//2:y//2,x//2:500] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)
equ = cv2.equalizeHist(img)
'''
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],mask,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
cv2.imshow('mask',mask)
cv2.imshow('masked_img',masked_img)
'''

clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(5,5)) 
cl1 = clahe.apply(img)

cv2.imshow('img',img)
cv2.imshow('equ',equ)
cv2.imshow('cl1',cl1)

k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    res = np.hstack((img,equ)) 
    cv2.imwrite('G:/a.jpg',res)
    
"""Size and rotation Invariants 
Descriptors will be same
mahotas and cv2 segmentation"""

