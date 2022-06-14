from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split as tts
import numpy as np
import argparse
import os, pickle
from Parameter_Calculation import Get_pickle, Get_Feature_Pickles

def baseline_model_1(activation, dropout, loss, optimizer, metrics):
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

def baseline_model_2(activation, dropout, loss, optimizer, metrics):
    # create model
    model = Sequential()
    shape = n_dim
    #40
    model.add(Dense(n_dim, input_dim=n_dim, use_bias=False, kernel_initializer='normal', activation=activation))
    shape = np.ceil(shape*1.7)
    shape = shape.astype(int)
    #68
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape*1.7)
    shape = shape.astype(int)
    #116
    model.add(Dense(shape, use_bias=False, activation=activation))
    model.add(Dropout(dropout))
    shape = np.ceil(shape*1.7)
    shape = shape.astype(int)
    #197
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/2)
    shape = shape.astype(int)
    #98
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/2)
    shape = shape.astype(int)
    #49
    model.add(Dense(shape, use_bias=False, activation=activation))
    model.add(Dropout(dropout))
    shape = np.ceil(shape/2)
    shape = shape.astype(int)
    #24
    model.add(Dense(shape, use_bias=False, activation=activation))
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss = loss, optimizer = optimizer, metrics = [metrics])
    return model

def baseline_model_3(activation, dropout, loss, optimizer, metrics):
    # create model
    model = Sequential()
    shape = 300
    #The Sequential model is a linear stack of layers.
    model.add(Dense(n_dim, input_dim=n_dim, use_bias=False, kernel_initializer='normal', activation=activation))
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/1.3)
    shape = shape.astype(int)
    #225
    model.add(Dense(shape, use_bias=False, activation=activation))
    model.add(Dropout(dropout))
    shape = np.ceil(shape/1.3)
    shape = shape.astype(int)
    #170
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/1.3)
    shape = shape.astype(int)
    #128
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/1.3)
    shape = shape.astype(int)
    #96
    model.add(Dense(shape, use_bias=False, activation=activation))
    model.add(Dropout(dropout))
    shape = np.ceil(shape/1.3)
    shape = shape.astype(int)
    #72
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/1.3)
    shape = shape.astype(int)
    #54
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/1.3)
    shape = shape.astype(int)
    #40
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/1.3)
    shape = shape.astype(int)
    #30
    model.add(Dense(shape, use_bias=False, activation=activation))
    shape = np.ceil(shape/1.3)
    shape = shape.astype(int)
    #23
    model.add(Dense(shape, use_bias=False, activation=activation))
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss = loss, optimizer = optimizer, metrics = [metrics])
    return model
if __name__ == '__main__':
    loss = 'categorical_crossentropy'
    test_size = 0.3
    For_File = ''
    dropout = 0.3
    random_state = 100
    n_layer = 7
    file_path = 'C:/Datasets/Urban_Sound_Dataset/Pickles/Feature_Pickles/'
    ap = argparse.ArgumentParser()
    ap.add_argument("-F", "--feature_name", required=False,
        help = "Features")
    ap.add_argument("-e", "--epochs", required=False,
        help = "Features")
    ap.add_argument("-b", "--batch_size", required=False,
        help = "Features")
    ap.add_argument("-o", "--optimizer", required=False,
        help = "Features")
    ap.add_argument("-a", "--activation", required=False,
        help = "Features")
    ap.add_argument("-m", "--metrics", required=False,
        help = "Features")
    args = vars(ap.parse_args())
    epochs = int(args['epochs'])
    batch_size = int(args['batch_size'])
    optimizer = args['optimizer']
    metrics = args['metrics']
    activation = args['activation']
    fe = args['feature_name']
    Feature_Dict = Get_Feature_Pickles(file_path)
    fe_pik = Feature_Dict[fe]
    Labels = Get_pickle(path = 'C:/Datasets/Urban_Sound_Dataset/Pickles/One_Hot_Encoded_Labels.pickle')
    #exit()
    X_train, X_test, Y_train, Y_test = tts(fe_pik, Labels[0],test_size = test_size, random_state = random_state)
    n_dim = X_train.shape[1]
    n_classes = np.unique(Labels[1]).shape[0]
    print(n_dim, n_classes)
    model_1 = baseline_model_1(activation = activation, dropout = dropout, loss = loss, 
                               optimizer = optimizer, metrics = metrics)
    model_1.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = epochs, 
              batch_size = batch_size, verbose = 0)
    model_2 = baseline_model_2(activation = activation, dropout = dropout, loss = loss, 
                               optimizer = optimizer, metrics = metrics)
    model_2.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = epochs, 
              batch_size = batch_size, verbose = 0)
    model_3 = baseline_model_3(activation = activation, dropout = dropout, loss = loss, 
                               optimizer = optimizer, metrics = metrics)
    model_3.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = epochs, 
              batch_size = batch_size, verbose = 0)
    scores, x1 = model_1.evaluate(X_test, Y_test, verbose=0)
    mod_1_acc = (x1*100)
    scores, x2 = model_2.evaluate(X_test, Y_test, verbose=0)
    mod_2_acc = (x2*100)
    scores, x3 = model_3.evaluate(X_test, Y_test, verbose=0)
    mod_3_acc = (x3*100)
    For_File += '\n' 
    For_File += '{features},{epochs},{batch_size},{optimizer},{loss},{activation},{metrics},{mod_1_acc},{mod_2_acc},{mod_3_acc}'.format(features = fe, epochs = str(epochs), batch_size = str(batch_size),
                            optimizer = str(optimizer), activation = str(activation), 
                            metrics = str(metrics), mod_1_acc = str(mod_1_acc), mod_2_acc = str(mod_2_acc),
                            mod_3_acc = str(mod_3_acc),loss = str(loss))
    combos = open('C:/Datasets/Urban_Sound_Dataset/Pickles/Data1.csv','a')
    combos.write(For_File)
    combos.close()
    print('Written Successfully in file')
    # Final evaluation of the model
    try:
        if int(mod_1_acc) >= 85:
            model_name = 'C:/Datasets/Urban_Sound_Dataset/Models/Audio_MLP_epochs_' +str(epochs)+ '_batch_size_' +str(batch_size)+ '_loss_' +str(loss)+ '_optimizer_' +str(optimizer)+ '_metrics_' +str(metrics)+ '_n_layer_' +str(n_layer)+ '_random_state_' +str(random_state)+ '_test_size_' +str(test_size)+ '_dropout_' +str(dropout)+ '_activation_'  +activation+ '_features_' +str(fe)+ '_suffix_' +'model_1'
            # serialize model to JSON
            model_json = model_1.to_json()
            json_file = open(model_name + '.json', "w")
            json_file.write(model_json)
            json_file.close()
            # serialize weights to HDF5
            model_1.save_weights(model_name + ".h5")
            print("Saved model to disk as   ",model_name)
        if int(mod_2_acc) >= 85:
            model_name = 'C:/Datasets/Urban_Sound_Dataset/Models/Audio_MLP_epochs_' +str(epochs)+ '_batch_size_' +str(batch_size)+ '_loss_' +str(loss)+ '_optimizer_' +str(optimizer)+ '_metrics_' +str(metrics)+ '_n_layer_' +str(n_layer)+ '_random_state_' +str(random_state)+ '_test_size_' +str(test_size)+ '_dropout_' +str(dropout)+ '_activation_'  +activation+ '_features_' +str(fe)+ '_suffix_' +'model_2'
            # serialize model to JSON
            model_json = model_2.to_json()
            json_file = open(model_name + '.json', "w")
            json_file.write(model_json)
            json_file.close()
            # serialize weights to HDF5
            model_2.save_weights(model_name + ".h5")
            print("Saved model to disk as   ",model_name)
        if int(mod_3_acc) >= 85:
            model_name = 'C:/Datasets/Urban_Sound_Dataset/Models/Audio_MLP_epochs_' +str(epochs)+ '_batch_size_' +str(batch_size)+ '_loss_' +str(loss)+ '_optimizer_' +str(optimizer)+ '_metrics_' +str(metrics)+ '_n_layer_' +str(n_layer)+ '_random_state_' +str(random_state)+ '_test_size_' +str(test_size)+ '_dropout_' +str(dropout)+ '_activation_'  +activation+ '_features_' +str(fe)+ '_suffix_' +'model_3'
            # serialize model to JSON
            model_json = model_3.to_json()
            with open(model_name + '.json', "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model_3.save_weights(model_name + ".h5")
            print("Saved model to disk as   ",model_name)
    except:
        print("Model can't be saved ! ! !")
    #combos.write('features,epochs,batch_size,optimizer,loss,activation,metrics,mod_1_acc,mod_2_acc,mod_3_acc')