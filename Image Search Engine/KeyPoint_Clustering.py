import cv2
import pickle
import numpy as np
import os

from sklearn.cluster import MiniBatchKMeans

def Load_Pickle():
    pickle_in = open('G:/Image Search Engine/Image_Descriptor_Pickle/Total_Descriptors.pickle','rb')
    Total_Descriptors = pickle.load(pickle_in) 
    pickle_in.close()
    return Total_Descriptors

if __name__ == '__main__':
    clusters = 1000
    rng = np.random.RandomState(0)
    kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=rng, verbose=True)
    Total_Descriptors = np.float32(Load_Pickle())
    count = len(Total_Descriptors)
    sample = 0
    while((sample + 1000) < count):
        data = Total_Descriptors[sample: sample + 1000]
        sample = sample + 1000
        kmeans.partial_fit(data)
        print(sample)
    kmeans.partial_fit(Total_Descriptors[sample: count])
    print(kmeans.cluster_centers_)
    import pandas as pd
    df = pd.DataFrame(kmeans.cluster_centers_)
    df.to_csv('G:/Image Search Engine/Image_Descriptor_Pickle/Centroids.csv')
