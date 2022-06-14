import numpy as np
import pickle
import os
from itertools import product as pro
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
def Get_Feature_Pickles():
    PickleFolder = os.listdir(file_path)
    PickleFiles = []
    Feature_Dict = {}
    for i in PickleFolder:
        PickleFiles.append(Get_pickle(path = file_path + i))
    for i in range(16):
        Feature_Dict[PickleFolder[i]] = PickleFiles[i]
    return Feature_Dict
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
    Feature_Dict = Get_Feature_Pickles()
    for fe,epochs,batch_size,optimizer,activation,metrics in required_combos:
        fe_pik = Feature_Dict[fe]
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
        scores, x1 = model_1.evaluate(X_test, Y_test, verbose=2)
        mod_1_acc = (x1*100)
        scores, x2 = model_2.evaluate(X_test, Y_test, verbose=2)
        mod_2_acc = (x2*100)
        scores, x3 = model_3.evaluate(X_test, Y_test, verbose=2)
        mod_3_acc = (x3*100)
        For_File += '\n' 
        For_File += '{features},{epochs},{batch_size},{optimizer},{activation},{metrics},{mod_1_acc},{mod_2_acc},{mod_3_acc}'.format(features = fe, epochs = str(epochs), batch_size = str(batch_size),
                                optimizer = str(optimizer), activation = str(activation), 
                                metrics = str(metrics), mod_1_acc = str(mod_1_acc), mod_2_acc = str(mod_2_acc),
                                mod_3_acc = str(mod_3_acc))
        # Final evaluation of the model
        if int(mod_1_acc) >= 85:
            model_name = 'C:/Datasets/Urban_Sound_Dataset/Models/Audio_MLP_epochs_' +str(epochs)+ '_batch_size_' +str(batch_size)+ '_loss_' +str(loss)+ '_optimizer_' +str(optimizer)+ '_metrics_' +str(metrics[0])+ '_n_layer_' +str(n_layer)+ '_random_state_' +str(random_state)+ '_test_size_' +str(test_size)+ '_dropout_' +str(dropout)+ '_activation_'  +activation+ '_suffix_' +'model_1'
            # serialize model to JSON
            model_json = model_1.to_json()
            with open(model_name + '.json', "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model_1.save_weights(model_name + ".h5")
            print("Saved model to disk as   ",model_name)
        if int(mod_2_acc) >= 85:
            model_name = 'C:/Datasets/Urban_Sound_Dataset/Models/Audio_MLP_epochs_' +str(epochs)+ '_batch_size_' +str(batch_size)+ '_loss_' +str(loss)+ '_optimizer_' +str(optimizer)+ '_metrics_' +str(metrics[0])+ '_n_layer_' +str(n_layer)+ '_random_state_' +str(random_state)+ '_test_size_' +str(test_size)+ '_dropout_' +str(dropout)+ '_activation_'  +activation+ '_suffix_' +'model_2'
            # serialize model to JSON
            model_json = model_2.to_json()
            with open(model_name + '.json', "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model_2.save_weights(model_name + ".h5")
            print("Saved model to disk as   ",model_name)
        if int(mod_3_acc) >= 85:
            model_name = 'C:/Datasets/Urban_Sound_Dataset/Models/Audio_MLP_epochs_' +str(epochs)+ '_batch_size_' +str(batch_size)+ '_loss_' +str(loss)+ '_optimizer_' +str(optimizer)+ '_metrics_' +str(metrics[0])+ '_n_layer_' +str(n_layer)+ '_random_state_' +str(random_state)+ '_test_size_' +str(test_size)+ '_dropout_' +str(dropout)+ '_activation_'  +activation+ '_suffix_' +'model_3'
            # serialize model to JSON
            model_json = model_3.to_json()
            with open(model_name + '.json', "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model_3.save_weights(model_name + ".h5")
            print("Saved model to disk as   ",model_name)
        count += 1
        print(count)
        if count == 5:
            break
    Create_pickle(path = file_path + 'temp.pickle', param = For_File)
    combos = open('C:/Datasets/Urban_Sound_Dataset/Pickles/Data1.csv','a')
    combos.write('features,epochs,batch_size,optimizer,activation,metrics,mod_1_acc,mod_2_acc,mod_3_acc')
    combos.write(For_File)
    combos.close()