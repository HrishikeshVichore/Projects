import pickle
import os
import shutil
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split as tts
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')

def GetShape(ShapeDirectory):
    df = pd.read_csv(ShapeDirectory,',')
    TotalImages = df.Nimages.sum()
    Xshape = (TotalImages,image_size,image_size,1)
    Yshape = (TotalImages,1)
    return Xshape, Yshape
def CreatePickle(ImageDirectory,Shape,PickleDirectory):

    for j in range(86):
        '''
        if j < 9:
            folder = 'Sample00' + str(j+1)
            print('Folder = ',folder)
        elif j >= 9: 
            folder = 'Sample0' + str(j+1)
            print('Folder = ',folder)
        '''
        folder = j
        Label  = j
        print('Folder = ',folder)
        dirPath = ImageDirectory + str(folder)
        #Nimages = len(os.listdir(dirPath))
        Nimages = 6
        Shape.write('\n{Nimages},{Label}'.format(Nimages = Nimages,Label = Label))
        dataset = np.ndarray(shape=(Nimages,image_size,image_size,1),dtype=np.float32)
        i = 0
        pickel_out = open(PickleDirectory +'/'+ str(Label) + '.pickle','wb')
        for file in range(14,25,2):
        #for file in os.listdir(dirPath):
            img = cv2.resize(cv2.imread(dirPath + '/' + str(file)+'.png',0),(image_size,image_size))
            #img = cv2.resize(cv2.imread(dirPath + '/' + file,0),(image_size,image_size))
            th,img = cv2.threshold(img,30,255,cv2.THRESH_BINARY)
            dataset[i,:,:,0] = img
            i = i + 1
        pickle.dump(dataset, pickel_out)
        pickel_out.close()
    Shape.close()
        
def load_dataset(TotalFiles,PickleDirectory):
    Xshape, Yshape = GetShape(ShapeDirectory)
    X = np.ndarray(shape = Xshape,dtype = np.float32)
    Y = np.ndarray(shape = Yshape,dtype = np.float32)
    Start = 0
    End   = 0
    for file in range(len(TotalFiles)-1) :
        WantedFile = PickleDirectory + '/' + TotalFiles[file]
        pickle_in = open(WantedFile,'rb')
        pickle_in = pickle.load(pickle_in)
        Label = TotalFiles[file].split('.')[0]
        End = End + len(pickle_in)
        X[Start:End,:,:,:] = pickle_in
        Y[Start:End,0] = Label
        Start = End
    X = X.reshape(X.shape[0] , 1, image_size, image_size).astype('float32')
    from scipy import stats
    #X = stats.zscore(X)
    X = X / 255
    Y = np_utils.to_categorical(Y)    
    return X, Y

def simple_model(num_classes):
    model = Sequential()
    model.add(Conv2D(16, (matrix_size, matrix_size), input_shape=(1, image_size, image_size), activation = activation))    
    '''model.add(ZeroPadding2D(padding = (pad_size, pad_size))) 
    model.add(Conv2D(16, (matrix_size, matrix_size), activation = activation))    
    model.add(ZeroPadding2D(padding = (pad_size, pad_size)))
    model.add(Conv2D(8, (matrix_size, matrix_size), activation = activation))
    model.add(ZeroPadding2D(padding = (pad_size, pad_size)))    
    model.add(Conv2D(8, (matrix_size, matrix_size), activation = activation))
    model.add(ZeroPadding2D(padding = (pad_size, pad_size)))
    model.add(Conv2D(8, (matrix_size, matrix_size), activation = activation))
    model.add(MaxPooling2D(pool_size=(pooling_size, pooling_size)))
    model.add(Conv2D(4, (matrix_size, matrix_size), activation = activation))'''
    model.add(MaxPooling2D(pool_size = (pooling_size, pooling_size)))
    model.add(Dropout(dropout))
    
    model.add(Flatten())
    model.add(Dense(240, activation = activation))
    model.add(Dense(120, activation = activation))
    model.add(Dense(num_classes, activation = FinalActivation))
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    return model

if __name__ == '__main__':
    #seed = 80
    epochs = 10
    matrix_size = 3
    pooling_size = 2
    dropout = 0.3
    pad_size = 1
    batch_size = 3
    image_size = 45
    random_state = 3
    test_size = 0.2
    prefix = 'Calibri'
    loss = 'categorical_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']
    activation = 'relu'
    FinalActivation = 'softmax'
    model_name = prefix + '_epochs_' + str(epochs) + '_batch_size_' + str(batch_size) + '_image_size_' + str(image_size) + '_random_state_' + str(random_state) + '_test_size_' + str(test_size)+ '_matrix_size_' + str(matrix_size) + '_pooling_size_' + str(pooling_size) + '_pad_size_' + str(pad_size) + '_dropout_' + str(dropout)
    np.seterr(divide='ignore', invalid='ignore')
    ImageDirectory = "G:/OCR/Dataset/Calibri Chars Dataset/"
    PickleDirectory = "G:/OCR/Pickles/CNN_Pickle 86 Chars Calibri"
    ShapeDirectory  = "G:/OCR/Pickles/CNN_Pickle 86 Chars Calibri/shape.csv" 
    #np.random.seed(seed)
    if os.path.isfile(ShapeDirectory):
        shutil.rmtree(PickleDirectory,ignore_errors = True)
        
    os.makedirs(name = PickleDirectory)
    Shape = open(ShapeDirectory,'a+')
    Shape.write('Nimages,Label,\n')
    CreatePickle(ImageDirectory, Shape,PickleDirectory)
    TotalFiles = os.listdir(PickleDirectory)
    X, Y = load_dataset(TotalFiles,PickleDirectory)
    
    X_train, X_test, Y_train, Y_test = tts(X, Y, test_size = test_size, random_state = random_state)
    num_classes = Y_test.shape[1]
    print('X_train = ',X_train.shape,'X_test = ',X_test.shape,
          'Y_train = ',Y_train.shape,'Y_test = ',Y_test.shape)
    model = simple_model(num_classes)

    model.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs = epochs, batch_size = batch_size, verbose = 2)    
    scores = model.evaluate(X_test, Y_test, verbose = 2)
    Accuracy = (scores[1]*100)
    print("Large MLP Accuracy = ",Accuracy)
    

    if int(Accuracy) >= 80:
        # serialize model to JSON
        model_json = model.to_json()
        with open(model_name + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_name + ".h5")
        print("Saved model to disk as   ",model_name)
        #exec(open("GetPoint1.py").read())
    