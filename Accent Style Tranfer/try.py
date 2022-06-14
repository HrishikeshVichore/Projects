import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization as BN, ReLU, MaxPool2D, Flatten, Dense
from tensorflow.keras.layers import Bidirectional, LSTM, Embedding, Reshape
from tensorflow.keras import Input
from MyCallback import MyCallback
from tensorflow.keras import Model

def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, n_fft=2048)
    p = np.angle(S)
    return np.log1p(np.abs(S[np.newaxis,:,:430])), fs

CONTENT_FILENAME = 'Audio_data_Wav/us_english/wallpaper.wav'
STYLE_FILENAME = 'Audio_data_Wav/indian_english/wallpaper.wav'

# Read both style and content
a_content, fs = read_audio_spectum(CONTENT_FILENAME)
a_style, fs = read_audio_spectum(STYLE_FILENAME)

N_SAMPLES = min(a_content.shape[2], a_style.shape[2])
N_CHANNELS = a_content.shape[1]
#N_CHANNELS = 160
a_style = a_style[:, :N_CHANNELS, :N_SAMPLES]
a_content = a_content[:, :N_CHANNELS, :N_SAMPLES]
# print(a_content.shape)
# print(a_style.shape)

# Display the spectograms

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('us_english')
# plt.imshow(a_content[0,:400,:])
# plt.subplot(1, 2, 2)
# plt.title('indian_english')
# plt.imshow(a_style[0,:400,:])
# plt.show()

N_FILTERS = 3
inputs = Input((1,N_CHANNELS,N_SAMPLES))
# print(inputs.shape)
hl = Conv2D(N_FILTERS, 3, strides=2, activation='relu', padding='same')(inputs)
# print(hl.shape)
hl = BN()(hl)
hl = ReLU()(hl)

hl = Conv2D(N_FILTERS, 3, strides=2, activation='relu', padding='same')(hl)
hl = BN()(hl)
hl = ReLU()(hl)
# print(hl.shape)
hl = MaxPool2D()(hl)
# print(hl.shape)
# hl = Flatten()(hl)
# print(hl.shape)
hl = Reshape((hl.shape[1]*hl.shape[2],hl.shape[3]))(hl)
# print(hl.shape)
op = Dense(1024, activation='relu')(hl)
# print(op.shape)
tf.keras.backend.set_image_data_format('channels_last')
hl = Bidirectional(LSTM(128, return_sequences=True))(op)
hl = Bidirectional(LSTM(128, return_sequences=True))(hl)
hl = Bidirectional(LSTM(128, return_sequences=True))(hl)
hl = Flatten()(hl)
hl = Dense(1024, activation='relu')(hl)
outputs = Dense(N_CHANNELS * N_SAMPLES,activation = 'relu')(hl)
outputs = Reshape((1, N_CHANNELS , N_SAMPLES))(outputs)

# outputs = Dense(16384,activation = 'relu')(hl)
# outputs = Reshape((1024, 128, 128))(outputs)
# print(outputs.shape)

model = Model(inputs = inputs, outputs = outputs)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
callback = MyCallback()
num_epochs = 1000

a_content_tf = np.ascontiguousarray(a_content[None,:,:])
a_style_tf = np.ascontiguousarray(a_style[None,:,:])

# print(model.summary())
# print(a_content_tf.shape)
# print(a_style_tf.shape)
history = model.fit(a_content_tf, a_style_tf, epochs=num_epochs, callbacks=[callback], verbose=0)

prediction = model.predict(a_content_tf)
# print(prediction.shape)
# print(type(prediction))
# print(prediction)
np.save('prediction.npy',prediction, allow_pickle=True)





