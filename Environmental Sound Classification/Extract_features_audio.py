import librosa as L
import os
import pandas as pd
from natsort import natsorted as Sort
import numpy as np
from librosa.feature import mfcc, chroma_stft, melspectrogram, spectral_contrast, tonnetz
from librosa.effects import harmonic 
import pickle
from librosa import stft



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
    return list(mfccs),list(chroma),list(mel),list(contrast),list(tonnetz_feature)

if __name__ == '__main__':
    file_path = 'D:/Datasets/Urban_Sound_Dataset/Pickles/'
    n_mfcc = 40
    dct_type = 2
    Stored_Pickle_Name = 'D:/Datasets/Urban_Sound_Dataset/Pickles/Audio_MLP_train' + '_n_mfcc_' + str(n_mfcc) +'_dct_type_' + str(dct_type) + '.pickle'
    mfcc_csv = file_path + 'mfcc.csv'
    chroma_csv = file_path + 'chroma.csv'
    mel_csv = file_path + 'mel.csv'
    contrast_csv = file_path + 'contrast.csv'
    tonnetz_csv = file_path + 'tonnetz.csv'
    
    pickle_in = Get_pickle(Stored_Pickle_Name)
    
    mfcc_file = open(mfcc_csv, 'a')
    chroma_file = open(chroma_csv,'a')
    mel_file = open(mel_csv,'a')
    contrast_file = open(contrast_csv,'a')
    tonnetz_file = open(tonnetz_csv,'a')
    mfcc_file.write('Features\tLabel')
    mfcc_file.write('\n')
    chroma_file.write('Features\tLabel')
    chroma_file.write('\n')
    mel_file.write('Features\tLabel')
    mel_file.write('\n')
    contrast_file.write('Features\tLabel')
    contrast_file.write('\n')
    tonnetz_file.write('Features\tLabel')
    tonnetz_file.write('\n')
    #count = 0
    
    for fn, label in zip(pickle_in[0], pickle_in[1]):
        mfccs, chroma, mel, contrast,tonnetz_feature = extract_feature(fn)
        mfcc_file.write('{feat}\t{label}'.format(feat = mfccs, label=label))
        mfcc_file.write('\n')
        chroma_file.write('{feat}\t{label}'.format(feat = chroma, label=label))
        chroma_file.write('\n')
        mel_file.write('{feat}\t{label}'.format(feat = mel, label=label))
        mel_file.write('\n')
        contrast_file.write('{feat}\t{label}'.format(feat = contrast, label=label))
        contrast_file.write('\n')
        tonnetz_file.write('{feat}\t{label}'.format(feat = tonnetz_feature, label=label))
        tonnetz_file.write('\n')
        #count += 1
        #print(count)
        
    mfcc_file.close()
    chroma_file.close()
    mel_file.close()
    contrast_file.close()
    tonnetz_file.close()
    print("Done")