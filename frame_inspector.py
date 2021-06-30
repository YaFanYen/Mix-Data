import os, glob

face_path = '/home/ed716/Documents/NewSSD/Cocktail/face_input'
face = os.listdir(face_path)

inspect_dir = 'face_input'
inspect_range = (0,len(face))
valid_frame_path = 'valid_frame.txt'

def check_frame(idx,dir=inspect_dir):
    path = dir + '/' + face[idx]
    if(not os.path.exists(path)): return False
    return True

for i in range(inspect_range[0],inspect_range[1]):
    valid = True
    if(check_frame(i)==False):
        path = inspect_dir + '/' + face[i]
        for file in glob.glob(path):
            os.remove(file)
        valid = False
        print('frame %s is not valid'%i)
        break
    if valid:
        with open(valid_frame_path,'a') as f:
            name = face[i]
            frame_name = name[:31]
            f.write(frame_name+'\n')
