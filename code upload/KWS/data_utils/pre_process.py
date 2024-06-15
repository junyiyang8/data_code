import numpy as np
import os
from tqdm import tqdm
TRAIN_DATA_DIR='audio/train'
TEST_DATA_DIR='audio/test'
VALIDATION_DATA_DIR='audio/validation'

def get_mean_std(data_dir=TRAIN_DATA_DIR):
    mean_list=[]
    std_list=[]
    for i in tqdm(os.listdir(data_dir)):
        data=np.load(data_dir+'/'+i,allow_pickle=True).tolist()
        feature=data['feature']
        mean_list.append(np.mean(feature,axis=0))
        std_list.append(np.std(feature,axis=0))
    mean=np.mean(np.array(mean_list),axis=0)
    std=np.mean(np.array(std_list),axis=0)
    return mean,std

def pre_process(mean,std,data_dir=TRAIN_DATA_DIR):
    for i in tqdm(os.listdir(data_dir)):
        data=np.load(data_dir+'/'+i,allow_pickle=True).tolist()
        data['feature']=(data['feature']-mean)/std
        np.save((data_dir+'/'+i), data)


# data=np.load('audio_nomean/test/30.npy',allow_pickle=True).tolist()
# print(data['feature'][0])
# # mean,std=get_mean_std()
# print('mean=',mean)
# print('std=',std) 
# pre_process(mean,std,TRAIN_DATA_DIR)
# pre_process(mean,std,TEST_DATA_DIR)
# pre_process(mean,std,VALIDATION_DATA_DIR)
# data=np.load('audio/test/30.npy',allow_pickle=True).tolist()
# print(data['feature'][0])

a=[[1,2,3],[4,5,6]]
k1 = np.transpose(a)
k2 = np.transpose(a)[1:-1]
print(k1)
print(k2)