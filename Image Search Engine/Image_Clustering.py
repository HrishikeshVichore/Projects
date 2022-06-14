import cv2
import pickle
import numpy as np
import os

def Kmeans_Cluster():
    print('a')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 2.0)
    print('b')
    ret,label,center = cv2.kmeans(Total_Descriptors,cluster,None,criteria,3,cv2.KMEANS_RANDOM_CENTERS)
    print('c')
    return ret,label,center
def Load_Pickle():
    pickle_in = open('Total_Descriptors.pickle','rb')
    Total_Descriptors = pickle.load(pickle_in) 
    pickle_in.close()
    return Total_Descriptors
if __name__ == '__main__':
    cluster = 1000
    Total_Descriptors = np.float32(Load_Pickle())
    print(len(Total_Descriptors))
    ret,label,center = Kmeans_Cluster()
    print(center)
        