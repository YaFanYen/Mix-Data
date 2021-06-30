from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import datetime
sys.path.append("../lib")
import AVHandler as avh
import pandas as pd

data_path = '/home/ed716/Documents/NewSSD/Cocktail/audio/norm_audio_train'
data = os.listdir(data_path)

def generate_frames(loc,start_idx,end_idx):
    # get frames for each video clip
    # loc        | the location of video clip
    # start_idx  | the starting index of the training sample
    # end_idx    | the ending index of the training sample

    avh.mkdir('frames')
    command = 'cd %s;' % loc
    for i in range(start_idx, end_idx):
        name = data[i]
        f_name = str(name[0:-4])
        command += 'ffmpeg -i %s.mp4 -pix_fmt yuvj422p -vf fps=25 ../../Cocktail/frames/%s-%%02d.jpg;' % (f_name, f_name)
        os.system(command)

generate_frames(loc='../Voxceleb/video', start_idx=0, end_idx=1000)
