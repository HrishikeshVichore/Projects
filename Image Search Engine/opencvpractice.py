import cv2,numpy as np,matplotlib.pyplot as plt

path_to_image = 'E:\\image.jpg'
img = cv2.imread(path_to_image)
"""
You can access a pixel value by its row and column coordinates. 
For BGR image, it returns an array of Blue, Green, Red values.
For grayscale image, just corresponding intensity is returned.
img[100,100] = [255,255,255]
Accessing Red Value
print(img.item(10,10,2))
Modifying Red values
img.itemset((10,10,2),100)
print(img.item(10,10,2))
print(img.shape)
#returns a tuple of number of rows, columns and channels (if image is color)
print(img.size)
#total number of pixels
print(img.dtype)
#datatype of img
"""
# 20x20 copied will be pasted as 20x20 only, no more nor less
#b,g,r = cv2.split(img)
#m = cv2.merge((b,r,r))
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
row, col = img.shape[:2]
res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
a = res[100:180,180:260]
res[200:280,250:330] = a
#M = np.float32([[1,0,100],[0,1,60],])
#dst = cv2.warpAffine(img,M,(col,row))
#cv2.imshow('image',dst)
cv2.imshow('image',img)
"""
Third argument of the cv.warpAffine() function is the size of the output image, 
which should be in the form of **(width, height)**. 
Remember width = number of columns, and height = number of rows.
"""
cv2.imshow('Magini',res)
#cv2.imshow('Merged',m)

M = cv2.getRotationMatrix2D((col/2,row/2),90,2)
dst = cv2.warpAffine(img,M,(col,row))
cv2.imshow('Rotate',dst)
k = cv2.waitKey(0)
print(img.shape)
#print(res.shape)


"""
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
"""
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    """
    ord() function in Python. Given a string of length one, return an integer representing the Unicode code
    point of the character when the argument is a unicode object, or the value of the byte when the 
    argument is an 8-bit string
    """
    cv2.imwrite('Conangray.png',img)
    cv2.destroyAllWindows()
    