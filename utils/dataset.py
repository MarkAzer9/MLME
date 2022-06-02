import cv2
import numpy as np
import os

def read_dataset(data_type,dim):
    img_count=0
    data_set=[]
    labels=[]
    root=os.path.join('./data/', data_type)


    for participant in os.listdir(root):
        participant_dir = os.path.join(root, participant)
        if os.path.isfile(participant_dir):
            continue
        for symbol in os.listdir(participant_dir):
            symbol_dir=os.path.join(participant_dir,symbol)
            if os.path.isfile(symbol_dir):
                continue
            for img in os.listdir(symbol_dir):
                img_path=os.path.join(symbol_dir, img)
                img=cv2.imread(img_path)
                gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                res=cv2.resize(gray, dsize=(dim, dim), interpolation=cv2.INTER_CUBIC)
                img_count=img_count+1
                print(img_count)
                data_set.append(res)
                labels.append(int(symbol))

    data=np.array(data_set)/255
    labels=np.array(labels)
    return data,labels

def flatten(data):
    return np.reshape(data,[-1,data.shape[1]*data.shape[2]])
