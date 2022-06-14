import torch
from librosa import load
from utils import wav2spectrum, spectrum2wav
from torch.autograd import Variable


cuda = True if torch.cuda.is_available() else False
CONTENT_FILENAME = "D:/Documents/Liclipse Workspace/Accent_Style_Transfer_Project_version_1/Recordings/Recording_01_14_2022__14_41_20.wav"
# STYLE_FILENAME = 'Audio_Data_Wav/us_english/abacus.wav'

a_content, sr = load(CONTENT_FILENAME)
# a_style, sr = load(STYLE_FILENAME)

a_content = wav2spectrum(y = a_content)
# a_style = wav2spectrum(y = a_style)

# N_SAMPLES = min(a_content.shape[1], a_style.shape[1])
# N_CHANNELS = a_content.shape[0]
# a_style = a_style[:N_CHANNELS, :N_SAMPLES]
# a_content = a_content[:N_CHANNELS, :N_SAMPLES]

a_content_torch = torch.from_numpy(a_content)[None, None, :, :]
if cuda:
    a_content_torch = a_content_torch.cuda()
print(a_content_torch.shape)

# a_style_torch = torch.from_numpy(a_style)[None, None, :, :]
# if cuda:
#     a_style_torch = a_style_torch.cuda()
# print(a_style_torch.shape)

model = torch.load('temp_model.pth')
model.eval()
print('Model Evaluated')

# a_C_var = Variable(a_content_torch, requires_grad=False).float()
# a_S_var = Variable(a_style_torch, requires_grad=False).float()
# if cuda:
#     model = model.cuda()
#     a_C_var = a_C_var.cuda()
#     a_S_var = a_S_var.cuda()
#
# a_C = model(a_C_var)
# a_S = model(a_S_var)
#
# print('Found Content and Style Features')

a_G_var = Variable(torch.randn(a_content_torch.shape) * 1e-3)
if cuda:
    a_G_var = a_G_var.cuda()
a_G_var.requires_grad = True
optimizer = torch.optim.Adam([a_G_var])

num_epochs = 1000
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    a_G = model(a_G_var)
    optimizer.step()

file_name = 'try_torch_train_load_model'
gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
gen_audio_C = f'output/audio/{file_name}.wav'
spectrum2wav(gen_spectrum, sr, gen_audio_C)
print('File saved')