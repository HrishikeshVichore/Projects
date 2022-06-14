import cv2
from cv2 import imread
from cv2 import xfeatures2d
import pandas as pd
import numpy as np
from scipy.spatial import distance
import pickle
import threading
import math 
import os

def gen_sift_features(gray_img):
    sift = xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def getClusters(start, end):
    #print(start, '      ', end)
    for i in range(start, end):
        if not os.path.isfile(dirPath + str(i) + '_new.pickle'):
            #countA += 1
            PicklePath = dirPath + str(i) + '.pickle'
            desc = pickle.load(open(PicklePath,'rb'))
            feature = [0] * vectorLength
            #print(len(desc))
            for dpt in desc:
                dpt = np.asarray(dpt)
                min = distance.cityblock(centroids[0], dpt)
                minLoc = 0
                for j in range (1, vectorLength):
                    p = distance.cityblock(centroids[j], dpt)
                    if(p < min):
                        min = p
                        minLoc = j
                feature[minLoc] = feature[minLoc] + 1
            print("Saved File : ", dirPath + str(i) + '_new.pickle')
            #print("Count = ", countA)
            
            file = open(dirPath + str(i) + '_new.pickle', 'wb')
            pickle.dump(feature, file)
            file.close()
        else:
            #countB += 1
            print('Already Done: ', dirPath + str(i) + '_new.pickle')
            #print("Count = ", countB)

if __name__ == '__main__':
    dirPath = 'G:/Image Search Engine/Image_Descriptor_Pickle/'
    Total_Descriptors = []
    df = pd.read_csv(dirPath + 'Centroids.csv')
    centroids = np.asarray(df.values)
    centroids = centroids[:, 1:centroids.shape[1]]
    print(centroids.shape)
    vectorLength = centroids.shape[0]
    
    thread_count = 12
    image_count = 1000
    thread_list = []
    #countA = 0
    #countB = 0
    for i in range(thread_count):
        start = math.floor(i * image_count / thread_count) + 1
        if i == 0:
            start = 0
        end = math.floor((i + 1) * image_count / thread_count) + 1
        if end >= 1000:
            end = 1000
        thread_list.append(threading.Thread(target=getClusters, args=(start, end)))
    
    for thread in thread_list:
        thread.start()
    
    for thread in thread_list:
        thread.join()
    