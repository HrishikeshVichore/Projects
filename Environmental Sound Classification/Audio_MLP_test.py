import numpy as np
from keras.models import model_from_json
from Audio_MLP_try import model_name, loss, optimizer, metrics

# load json and create model
json_file = open(model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_name + ".h5")
loaded_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

print('Model loaded and compiled')


