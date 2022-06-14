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
    file_path += 'try/' 
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
    features, labels = np.empty((0,163)), np.empty(0)
    i = 0
    for fn, label in zip(pickle_in[0], pickle_in[1]):
            mfccs, chroma, mel, contrast,tonnetz_feature = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz_feature])
            print(i, label)
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label)
            i += 1
    return np.array(features), np.array(labels, dtype = str)

def one_hot_encode(labels):
    le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")
    X = np.array(labels)
    X = le.fit_transform(X)
    return X

def baseline_model():
    # create model
    model = Sequential()
    #The Sequential model is a linear stack of layers.
    model.add(Dense(n_dim, input_dim=n_dim, use_bias=False, kernel_initializer='normal', activation=activation))
    model.add(Dense(125, use_bias=False, activation=activation))
    model.add(Dense(100, use_bias=False, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(75, use_bias=False, activation=activation))
    model.add(Dense(55, use_bias=False, activation=activation))
    model.add(Dense(35, use_bias=False, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    return model

if __name__ == '__main__':
    #For Train
    file_path = 'D:/Datasets/Urban_Sound_Dataset/train/'
    n_mfcc = 10
    dct_type = 2
    test_size = 0.3
    random_state = 100
    epochs = 30
    batch_size = 100
    loss = 'categorical_crossentropy'
    optimizer = 'adadelta'
    activation = 'relu'
    metrics = ['categorical_accuracy']
    dropout = 0.3
    n_layer = 7
    model_name = 'D:/Datasets/Urban_Sound_Dataset/Models/Audio_MLP_epochs_' +str(epochs)+ '_batch_size_' +str(batch_size)+ '_loss_' +str(loss)+ '_optimizer_' +str(optimizer)+ '_metrics_' +str(metrics[0])+ '_n_layer_' +str(n_layer)+ '_random_state_' +str(random_state)+ '_test_size_' +str(test_size)+ '_dropout_' +str(dropout)+ '_activation_'  +activation
    Stored_Pickle_Name = 'D:/Datasets/Urban_Sound_Dataset/Pickles/Audio_MLP_train' + '_n_mfcc_' + str(n_mfcc) +'_dct_type_' + str(dct_type) + '.pickle'
    features_pickle = 'D:/Datasets/Urban_Sound_Dataset/Pickles/Features_pickle_train.pickle'
    
    '''#For one time use only comment afterwards 
    load_sound_files(file_path, Stored_Pickle_Name, 'train.csv')
    pickle_in = Get_pickle(Stored_Pickle_Name)
    features, original_labels = parse_audio_files(pickle_in)
    labels = one_hot_encode(original_labels)
    features_labels = [features, labels, original_labels]
    Create_pickle(features_pickle, features_labels)'''
    
    pickle_in_train = Get_pickle(features_pickle)
    print(pickle_in_train[0])
    exit()
    X_train, X_test, Y_train, Y_test = tts(pickle_in_train[0], pickle_in_train[1],
                                test_size = test_size, random_state = random_state)
        
    n_dim = pickle_in_train[0].shape[1]
    n_classes = np.unique(pickle_in_train[2]).shape[0]
    print('n_dim ' +str(n_dim)+ 'n_classes ' +str(n_classes))
    model = baseline_model()
    model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = epochs, 
              batch_size = batch_size, verbose = 2)
    # Final evaluation of the model
    scores, x = model.evaluate(X_test, Y_test, verbose=2)
    Accuracy = (scores*100)
    print("Large MLP Accuracy = ",Accuracy)
    
    if int(Accuracy) >= 80 and int(Accuracy) <= 100:
        # serialize model to JSON
        model_json = model.to_json()
        with open(model_name + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_name + ".h5")
        print("Saved model to disk as   ",model_name)
    