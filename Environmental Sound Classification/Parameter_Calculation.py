import numpy as np
import pickle
import os
from itertools import product as pro
import subprocess as sb
from time import sleep
import pandas as pd
def Get_pickle(path):
    pickle_in = open(path,'rb')
    pickle_in = pickle.load(pickle_in)
    return pickle_in

def Create_pickle(path,param):
    pickle_out = open(path , 'wb')
    pickle.dump(param, pickle_out)
    pickle_out.close()
    
def Get_Feature_Pickles(file_path):
    PickleFolder = os.listdir(file_path)
    PickleFiles = []
    Feature_Dict = {}
    for i in PickleFolder:
        PickleFiles.append(Get_pickle(path = file_path + i))
    for i in range(16):
        Feature_Dict[PickleFolder[i]] = PickleFiles[i]
    return Feature_Dict

if __name__ == '__main__':
    file_path = 'C:/Datasets/Urban_Sound_Dataset/Pickles/Feature_Pickles/'
    epochs = [20, 30, 40, 50]
    batch_size = [100, 150, 200]
    loss = 'categorical_crossentropy'
    optimizer = ['RMSprop', 'adagrad', 'adadelta', 'adam', 'nadam']
    activation = ['relu', 'sigmoid']
    metrics = ['accuracy', 'categorical_accuracy']
    test_size = 0.3
    dropout = 0.3
    random_state = 100
    n_layer = 7
    features = os.listdir(path = file_path)
    required_lists = [features,epochs, batch_size, optimizer, activation, metrics]
    required_combos = list(pro(*required_lists))
    Labels = Get_pickle(path = 'C:/Datasets/Urban_Sound_Dataset/Pickles/One_Hot_Encoded_Labels.pickle')
    For_File = ''
    count = 0
    df = pd.read_csv('remaining_combinations.csv')
    a = df['features']
    b = df['epochs']
    c = df['batch_size']
    d = df['optimizer']
    e = df['loss']
    f = df['activation']
    g = df['metrics']
    for fe,epochs,batch_size,optimizer,activation,metrics in zip(a,b,c,d,f,g):
        #print(fe,epochs,batch_size,optimizer,activation,metrics)
        param = 'python D:/My_Codes/Audio_CNN/Training_file.py -F {fe} -e {e} -b {bs} -o {o} -a {act} -m {m}'.format(
            fe = fe, e = epochs, bs = batch_size, o = optimizer, 
            act = activation, m = metrics)
        Model_Training_Process = sb.Popen(param , shell = True)
        Model_Training_Process.wait()
        sleep(3)
        count += 1
        print(count)
    sleep(3)
    os.system('shutdown -s')