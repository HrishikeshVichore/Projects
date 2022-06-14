from itertools import combinations as com
import pandas as pd
import numpy as np
import pickle 
from ast import literal_eval as le

def Create_pickle(path,param):
    pickle_out = open(path , 'wb')
    pickle.dump(param, pickle_out)
    pickle_out.close()
    
file_path = "D:/Datasets/Urban_Sound_Dataset/Pickles/"
feature_pickle_path = "D:/Datasets/Urban_Sound_Dataset/Pickles/Feature_Pickles/"

available_features = ['mfcc', 'chroma', 'mel', 'contrast', 'tonnetz'] 

x = []

for i in range(1,6): 
    x.append(list(com(available_features,i)))

feature_combos = []

for i in x:
    for j in i:
        if 'mfcc' in j:
            feature_combos.append(list(j))
print(len(feature_combos))

mfccs = pd.read_csv(file_path +'mfcc'+ '.csv', sep = '\t')
chroma = pd.read_csv(file_path +'chroma'+ '.csv', sep = '\t')
mel = pd.read_csv(file_path +'mel'+ '.csv', sep = '\t')
contrast = pd.read_csv(file_path +'contrast'+ '.csv', sep = '\t')
tonnetz = pd.read_csv(file_path +'tonnetz'+ '.csv', sep = '\t')

dict = {'mfcc':mfccs,
        'chroma':chroma,
        'mel':mel,
        'contrast':contrast,
        'tonnetz': tonnetz}
dict1 = {'mfcc':40,
         'chroma':12,
         'mel':128,
         'contrast':7,
         'tonnetz':6}
    
for i in feature_combos:
    if len(i) == 1:
        shape = dict1[i[0]]
        feature_1 = np.empty((0,shape))
        for feature in dict[i[0]]['Features']:
            feature = le(feature)
            feature = np.asarray(feature)
            print(shape)
            feature_1 = np.vstack([feature_1,np.hstack([feature])])
        path = feature_pickle_path +'1_'+ i[0]+ '.pickle'
        Create_pickle(path, param = feature_1) 
            
    if len(i) == 2:
        shape = dict1[i[0]] + dict1[i[1]]
        feature_2 = np.empty((0,shape))
        for k,l in zip(dict[i[0]]['Features'],dict[i[1]]['Features']):
            k = le(k)
            l = le(l)
            k = np.asarray(k)
            l = np.asarray(l)
            print(shape)
            feature_2 = np.vstack([feature_2,np.hstack([k,l])])
        path = feature_pickle_path+ '2_' +i[0]+ '_' +i[1]+ '.pickle'
        Create_pickle(path, param = feature_2) 
    
    if len(i) == 3:
        shape = dict1[i[0]] + dict1[i[1]] + dict1[i[2]]
        feature_3 = np.empty((0,shape))
        for k,l,m in zip(dict[i[0]]['Features'],dict[i[1]]['Features'],dict[i[2]]['Features']):
            k = le(k)
            l = le(l)
            m = le(m)
            k = np.asarray(k)
            l = np.asarray(l)
            m = np.asarray(m)          
            print(shape)
            feature_3 = np.vstack([feature_3,np.hstack([k,l,m])])
        path = feature_pickle_path+ '3_'  +i[0]+ '_' +i[1]+ '_' +i[2]+ '.pickle'
        Create_pickle(path, param = feature_3) 
        
    if len(i) == 4:
        shape = dict1[i[0]] + dict1[i[1]]+ dict1[i[2]] + dict1[i[3]]
        feature_4 = np.empty((0,shape))
        for k,l,m,n in zip(dict[i[0]]['Features'],dict[i[1]]['Features'],dict[i[2]]['Features'],dict[i[3]]['Features']):
            k = le(k)
            l = le(l)
            m = le(m)
            n = le(n)
            k = np.asarray(k)
            l = np.asarray(l)
            m = np.asarray(m)
            n = np.asarray(n)           
            print(shape)
            feature_4 = np.vstack([feature_4,np.hstack([k,l,m,n])])
        path = feature_pickle_path+ '4_'  +i[0]+ '_' +i[1]+ '_' +i[2]+ '_' +i[3]+ '.pickle'
        Create_pickle(path, param = feature_4) 
    
    if len(i) == 5:
        shape = dict1[i[0]] + dict1[i[1]]+ dict1[i[2]] + dict1[i[3]] + dict1[i[4]]
        feature_5 = np.empty((0,shape))
        for k,l,m,n,o in zip(dict[i[0]]['Features'],dict[i[1]]['Features'],dict[i[2]]['Features'],dict[i[3]]['Features'],dict[i[4]]['Features']):
            k = le(k)
            l = le(l)
            m = le(m)
            n = le(n)
            o = le(o)
            k = np.asarray(k)
            l = np.asarray(l)
            m = np.asarray(m)
            n = np.asarray(n)
            o = np.asarray(o)           
            print(shape)
            feature_5 = np.vstack([feature_5,np.hstack([k,l,m,n,o])])
        path = feature_pickle_path+ '5_'  +i[0]+ '_' +i[1]+ '_' +i[2]+ '_' +i[3]+ '_' +i[4]+ '.pickle'
        Create_pickle(path, param = feature_5) 
        