import numpy as np
from librosa import istft
from librosa.output import write_wav
from librosa import griffinlim
import librosa
from denoiser import reduce_noise_power as denoise, enhance

n_fft = 2048
sr = 22050

prediction = np.load('prediction.npy', allow_pickle=1)[0][0]
print(prediction.shape)

#spect, _ = magphase(prediction)
x = griffinlim(prediction, n_iter=1000)

write_wav(f'output/prediction_{n_fft}.wav', x, sr)

y, sr = librosa.load(f'output/prediction_{n_fft}.wav')

y = denoise(y, sr)

y = enhance(y)

write_wav(f'output/prediction_{n_fft}_cleaned.wav', y, sr)






