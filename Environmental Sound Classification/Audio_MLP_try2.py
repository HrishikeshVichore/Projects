import librosa as L
import os
import pandas as pd
from natsort import natsorted as Sort
import numpy as np
from librosa.feature import mfcc, chroma_stft, melspectrogram, spectral_contrast, tonnetz
from librosa.effects import harmonic 
import pickle
from librosa import stft
import category_encoders as ce
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split as tts
from itertools import product as pro


def Get_pickle(path):
    pickle_in = open(path,'rb')
    pickle_in = pickle.load(pickle_in)
    return pickle_in
def Create_pickle(path,param):
    pickle_out = open(path , 'wb')
    pickle.dump(param, pickle_out)
    pickle_out.close()
    
def load_sound_files(file_path, Stored_Pickle_Name, csv_name):
    raw_sounds = []
    df = pd.read_csv(file_path + csv_name)
    raw_sounds_Class = list(df['Class'])
    file_path += 'Train/' 
    for fp in Sort(os.listdir(file_path)):
        fp = file_path + fp
        print(fp)
        Audio,sr = L.load(path = fp)
        raw_sounds.append(Audio)
    For_Pickle = [raw_sounds, raw_sounds_Class]
    Create_pickle(Stored_Pickle_Name, For_Pickle)
    
def extract_feature(X):
    sample_rate = 22050
    stft_feature = np.abs(stft(X))
    mfccs = np.mean(mfcc(y=X, sr=sample_rate, n_mfcc = n_mfcc,dct_type = dct_type).T,axis=0)
    chroma = np.mean(chroma_stft(S=stft_feature, sr=sample_rate).T,axis=0)
    mel = np.mean(melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(spectral_contrast(S=stft_feature, sr=sample_rate).T,axis=0)
    tonnetz_feature = np.mean(tonnetz(y=harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz_feature

def parse_audio_files(pickle_in):
    features_1, features_2, features_3, features_4, features_5,labels = np.empty((0,40)), np.empty((0,52)), np.empty((0,180)), np.empty((0,187)), np.empty((0,193)),np.empty(0)
    i = 0
    for fn, label in zip(pickle_in[0], pickle_in[1]):
            mfccs, chroma, mel, contrast,tonnetz_feature = extract_feature(fn)
            mfccs_feature = np.hstack([mfccs])
            mfccs_chroma = np.hstack([mfccs,chroma])
            mfccs_chroma_mel = np.hstack([mfccs,chroma,mel])
            mfccs_chroma_mel_contrast = np.hstack([mfccs,chroma,mel,contrast])
            mfccs_chroma_mel_contrast_tonnetz = np.hstack([mfccs,chroma,mel,contrast,tonnetz_feature])
            print(i, label)
            print('mfccs_feature ', mfccs_feature.shape, 'mfccs_chroma', mfccs_chroma.shape,
                  'mfccs_chroma_mel', mfccs_chroma_mel.shape, 'mfccs_chroma_mel_contrast',
                  mfccs_chroma_mel_contrast.shape, 'mfccs_chroma_mel_contrast_tonnetz',
                  mfccs_chroma_mel_contrast_tonnetz.shape)
            features_1 = np.vstack([features_1,mfccs_feature])
            features_2 = np.vstack([features_2,mfccs_chroma])
            features_3 = np.vstack([features_3,mfccs_chroma_mel])
            features_4 = np.vstack([features_4,mfccs_chroma_mel_contrast])
            features_5 = np.vstack([features_5,mfccs_chroma_mel_contrast_tonnetz])
            
            labels = np.append(labels, label)
            i += 1
    return np.array(features_1),np.array(features_2),np.array(features_3),np.array(features_4),np.array(features_5),np.array(labels, dtype = str)

def one_hot_encode(labels):
    le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")
    X = np.array(labels)
    X = le.fit_transform(X)
    return X

def baseline_model(activation, dropout, loss, optimizer, metrics):
    # create model
    model = Sequential()
    shape = n_dim
    #The Sequential model is a linear stack of layers.
    model.add(Dense(n_dim, input_dim=n_dim, use_bias=False, kernel_initializer='normal', activation=activation))
    shape = np.ceil(shape/1.25)
    shape = shape.astype(int)
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/1.25)
    shape = shape.astype(int)
    model.add(Dense(shape, use_bias=False, activation=activation))
    model.add(Dropout(dropout))
    shape = np.ceil(shape/1.25)
    shape = shape.astype(int)
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/1.25)
    shape = shape.astype(int)
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/1.25)
    shape = shape.astype(int)
    model.add(Dense(shape, use_bias=False, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss = loss, optimizer = optimizer, metrics = [metrics])
    return model
if __name__ == '__main__':
    suffixes = ['mfcc','mfccs_chroma','mfccs_chroma_mel','mfccs_chroma_mel_contrast','mfccs_chroma_mel_contrast_tonnetz']
    count = 0
    combos = open('Data.csv','a')
    combos.write('features,epochs,batch_size,optimizer,activation,metrics,accuracy')
    for i,suffix in enumerate(suffixes):
        #For Train
        file_path = 'D:/Datasets/Urban_Sound_Dataset/train/'
        n_mfcc = 40
        dct_type = 2
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
        Stored_Pickle_Name = 'D:/Datasets/Urban_Sound_Dataset/Pickles/Audio_MLP_train' + '_n_mfcc_' + str(n_mfcc) +'_dct_type_' + str(dct_type) + '.pickle'
        features_pickle = 'D:/Datasets/Urban_Sound_Dataset/Pickles/Features_pickle_train_mfcc.pickle'
        required_lists = [epochs, batch_size, optimizer, activation, metrics]
        required_combos = list(pro(*required_lists))
        
        '''#For one time use only comment afterwards 
        load_sound_files(file_path, Stored_Pickle_Name, 'train.csv')
        pickle_in = Get_pickle(Stored_Pickle_Name)
        features_1, features_2, features_3, features_4, features_5, original_labels = parse_audio_files(pickle_in)
        labels = one_hot_encode(original_labels)
        features = [features_1, features_2, features_3, features_4, features_5]
        features_labels = [features, labels, original_labels]
        Create_pickle(features_pickle, features_labels)'''
        
        pickle_in_train = Get_pickle(features_pickle)
        X_train, X_test, Y_train, Y_test = tts(pickle_in_train[0][i], pickle_in_train[1],
                                    test_size = test_size, random_state = random_state)
            
        n_dim = pickle_in_train[0][i].shape[1]
        n_classes = np.unique(pickle_in_train[2]).shape[0]
        print('n_dim ' +str(n_dim)+ 'n_classes ' +str(n_classes), pickle_in_train[1].shape)
        
        exit()
        for epochs, batch_size, optimizer, activation, metrics in required_combos:
            model = baseline_model(activation = activation, dropout = dropout, loss = loss, 
                                   optimizer = optimizer, metrics = metrics)
            model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = epochs, 
                      batch_size = batch_size, verbose = 0)
            # Final evaluation of the model
            scores, x = model.evaluate(X_test, Y_test, verbose=2)
            Accuracy = (x*100)
            print("Large MLP Accuracy = ",Accuracy)
            combos.write('\n')
            combos.write('{features},{epochs},{batch_size},{optimizer},{activation},{metrics},{accuracy}'
                        .format(features = suffix, epochs = str(epochs), batch_size = str(batch_size),
                                optimizer = str(optimizer), activation = str(activation), 
                                metrics = str(metrics), accuracy = str(Accuracy)))
            model_name = 'D:/Datasets/Urban_Sound_Dataset/Models/Audio_MLP_epochs_' +str(epochs)+ '_batch_size_' +str(batch_size)+ '_loss_' +str(loss)+ '_optimizer_' +str(optimizer)+ '_metrics_' +str(metrics[0])+ '_n_layer_' +str(n_layer)+ '_random_state_' +str(random_state)+ '_test_size_' +str(test_size)+ '_dropout_' +str(dropout)+ '_activation_'  +activation+ '_suffix_' +suffix
            
            if int(Accuracy) >= 85 and int(Accuracy) <= 100:
                # serialize model to JSON
                model_json = model.to_json()
                with open(model_name + '.json', "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights(model_name + ".h5")
                print("Saved model to disk as   ",model_name)
            count += 1
            print(count)
    combos.close()