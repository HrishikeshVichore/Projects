import cv2
from cv2 import imread
import os
from cv2 import xfeatures2d
import pickle
import numpy as np

def Create_Pickle():
    pickle_out = open('G:/Image Search Engine/Image_Descriptor_Pickle/Total_Descriptors.pickle','wb')
    pickle.dump(Total_Descriptors,pickle_out)
    pickle_out.close()  
if __name__ == '__main__':
    Total_Descriptors = []
    dirPath = 'G:/Image Search Engine/test1'
    path_for_pickle = "G:/Image Search Engine/Image_Descriptor_Pickle/"
    #for i in range(len(os.listdir(dirPath))):
    for i in range(1000):
        ImgPath = dirPath + '/' + str(i) + '.jpg'
        img  = imread(ImgPath)
        pickle_name = path_for_pickle + str(i) + '.pickle'
        p = open(pickle_name,'wb')
        if(i%100 == 0):
            print(ImgPath)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(gray, None)
        pickle.dump(desc,p)
        p.close()
        for dpt in desc:
            Total_Descriptors.append(dpt)
          
    
    Total_Descriptors = np.asarray(Total_Descriptors) 
    print(Total_Descriptors.shape)
    
    Create_Pickle()
    
    print('Pickle Saved')