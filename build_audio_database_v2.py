import sys
sys.path.append("../../model/lib")
import os
import librosa
import numpy as np
import utils
import itertools
import time
import scipy.io.wavfile as wavfile
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_path = '/home/ed716/Documents/NewSSD/Cocktail/audio/norm_audio_train'
norm_data = os.listdir(data_path)

# Parameter
SAMPLE_RANGE = (0,len(norm_data)) # data usage to generate database
WAV_REPO_PATH = os.path.expanduser("norm_audio_train")
DATABASE_REPO_PATH = 'all_5spk'
FRAME_LOG_PATH = 'valid_frame.txt'
NUM_SPEAKER = 5
MAX_NUM_SAMPLE = 10000


# time measure decorator
def timit(func):
    def cal_time(*args,**kwargs):
        tic = time.time()
        result = func(*args,**kwargs)
        tac = time.time()
        return result
    return cal_time

# create directory to store database
def init_dir(path = DATABASE_REPO_PATH ):
    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isdir('%s/mix_wav'%path):
        os.mkdir('%s/mix_wav'%path)

@timit
def generate_path_list(sample_range=SAMPLE_RANGE,repo_path=WAV_REPO_PATH,frame_path=FRAME_LOG_PATH):
    '''
    :return: 2D array with idx and path (idx_wav,path_wav)
    '''
    audio_path_list = []
    frame_set = set()

    with open(frame_path,'r') as f:
        frames = f.readlines()
    
    for i in range(len(frames)):
        frame = frames[i].replace('\n','').replace('frame_','') + '.wav'
        frame_set.add(str(frame))

    for i in range(sample_range[0],sample_range[1]):
        path = repo_path + '/' + norm_data[i]
        if os.path.exists(path) and (norm_data[i] in frame_set):
            audio_path_list.append((i,path))
    return audio_path_list

# data generate function
def single_audio_to_npy(audio_path_list,database_repo=DATABASE_REPO_PATH,fix_sr=8000):
    for idx,path in audio_path_list:
        data, _ = librosa.load(path, sr=fix_sr)
        data = utils.fast_stft(data)
        name = 'single-' + str(path[-29:-4])


# split single TF data to different part in order to mix
def split_to_mix(audio_path_list, database_repo=DATABASE_REPO_PATH, partition=2):
    # return split_list : (part1,part2,...)
    # each part : (idx,path)
    length = len(audio_path_list)
    part_len = length // partition
    head = 0
    part_idx = 0
    split_list = []
    while((head+part_len)<length):
        part = audio_path_list[head:(head+part_len)]
        split_list.append(part)
        head = head + part_len
        part_idx = part_idx + 1
    return split_list

# mix single TF data
def all_mix(split_list,database_repo=DATABASE_REPO_PATH,partition=2):
    assert len(split_list) == partition
    print('mixing data...')
    num_mix = 1
    num_mix_check = 0
    for part in split_list:
        num_mix *= len(part)

    part_len = len(split_list[-1])
    idx_list = [x for x in range(part_len)]
    combo_idx_list = itertools.product(idx_list,repeat=partition)
    for combo_idx in combo_idx_list:
        num_mix_check +=1
        single_mix(combo_idx,split_list,database_repo)


# mix several wav file and store TF domain data with npy
def single_mix(combo_idx,split_list,database_repo):
    assert len(combo_idx) == len(split_list)
    mix_rate = 1.0 / float(len(split_list))
    wav_list = []
    prefix = 'mix-'
    mid_name = ''

    for part_idx in range(len(split_list)):
        idx,path = split_list[part_idx][combo_idx[part_idx]]
        wav, sr = librosa.load(path, sr=8000)
        wav_list.append(wav)
        mid_name += str(path[-29:-4])
    wav_list = pad_sequences(wav_list, maxlen = 80000, dtype='float32', padding = 'post')

    # mix wav file
    mix_wav = np.zeros_like(wav_list[0])
    for wav in wav_list:
        mix_wav = mix_wav + wav * mix_rate

    # save mix wav file
    wav_name = prefix + mid_name + '.wav'
    wavfile.write('%s/mix_wav/%s'%(database_repo,wav_name),8000,mix_wav)

    # transfer mix wav to TF domain
    F_mix = utils.fast_stft(mix_wav)
    name = prefix + mid_name + ".npy"
    store_path = '%s/mix/%s'%(database_repo,name)


if __name__ == "__main__":
    init_dir()
    audio_path_list = generate_path_list()
    single_audio_to_npy(audio_path_list)
    split_list = split_to_mix(audio_path_list,partition=NUM_SPEAKER)
    all_mix(split_list,partition=NUM_SPEAKER)
