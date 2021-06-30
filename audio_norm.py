import librosa
import os
import numpy as np
import scipy.io.wavfile as wavfile

RANGE = (0,2000)
data_path = '/home/ed716/Documents/NewSSD/Voxceleb/audio'
data = os.listdir(data_path)
files = []

for i in range(len(data)):
    files.append(data_path + "/" + data[i])

if(not os.path.isdir('norm_audio_train')):
    os.mkdir('norm_audio_train')

for num in range(RANGE[0],RANGE[1]):
    path = files[num]
    norm_path = 'norm_audio_train/' + data[num]
    if (os.path.exists(path)):
        audio,_= librosa.load(path,sr=8000)
        max = np.max(np.abs(audio))
        norm_audio = np.divide(audio,max)
        wavfile.write(norm_path,8000,norm_audio)

















