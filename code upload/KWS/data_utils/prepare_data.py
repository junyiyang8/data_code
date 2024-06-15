import librosa
import numpy as np
import os
import random
import pickle
from tqdm import trange

def process_google_sample(sample_path,frame_length = 0.04,frame_shift = 0.02,n_mfcc = 40):
    if(type(sample_path)==str):
        signal, sr = librosa.load(sample_path,sr=16000)
    else:
        signal=sample_path
        sr=16000
    if(len(signal)!=16000):
        # print(f'Drop sample {sample_path}, because the duration is {librosa.get_duration(y=signal, sr=sr)} seconds')
        return None

    frame_length_samples = int(frame_length * sr)
    frame_shift_samples = int(frame_shift * sr)

    signal=(signal-signal.mean())/signal.std()  

    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=2048, n_mfcc=n_mfcc, hop_length=frame_shift_samples,win_length=frame_length_samples,center=True,norm=None)
    mfccs = np.transpose(mfccs)[1:-1]
    print(mfccs[0,:])
    return mfccs

def prepare_google(data_dir="audio",save_dir="audio"):    

    for i in ['train','test','validation']:
        if(not os.path.exists(save_dir+'/'+i)):
            os.makedirs(save_dir+'/'+i)

    classes=['yes','no','up','down','left','right','on','off','stop','go']    
    unknown=['eight','nine', 'bed', 'five', 'dog', 'seven', 'three', 'marvin', 'wow',
    'house', 'sheila', 'two', 'six', 'cat', 'one', 'tree', 'four', 'happy', 'zero', 'bird']
    # silence='_background_noise_'

    f=open(data_dir+'/testing_list.txt','r')
    test_list=f.readlines()
    test_list=[i[:-1] for i in test_list]

    f=open(data_dir+'/validation_list.txt','r')
    validation_list=f.readlines()
    validation_list=[i[:-1] for i in validation_list]

    print('process classes')
    cnt=0
    for i in trange(len(classes)):

        class_name=classes[i]
        sample_dir=data_dir+'/'+class_name
        sample_list=os.listdir(sample_dir)
        for sample in sample_list:

            sample_path=sample_dir+'/'+sample
            mfccs=process_google_sample(sample_path)
            if(not(mfccs is None)):
                # continue
            # else:
                data={'feature':mfccs,'label':i}
                if(sample_path.split('/')[-2]+'/'+sample_path.split('/')[-1] in test_list):
                    np.save(save_dir+'/'+'test'+'/'+str(cnt)+'.npy',data)
                elif(sample_path.split('/')[-2]+'/'+sample_path.split('/')[-1] in validation_list):
                    np.save(save_dir+'/'+'validation'+'/'+str(cnt)+'.npy',data)
                else:
                    np.save(save_dir+'/'+'train'+'/'+str(cnt)+'.npy',data)
                cnt+=1
    
    print('process unknown')
    for i in trange(len(unknown)):
        class_name=unknown[i]
        sample_dir=data_dir+'/'+class_name
        sample_list=os.listdir(sample_dir)
        for sample in sample_list:
            sample_path=sample_dir+'/'+sample
            mfccs=process_google_sample(sample_path)
            if(not(mfccs is None)):
                # continue
            # else:
                data={'feature':mfccs,'label':10}
                if(sample_path.split('/')[-2]+'/'+sample_path.split('/')[-1] in test_list):
                    np.save(save_dir+'/'+'test'+'/'+str(cnt)+'.npy',data)
                elif(sample_path.split('/')[-2]+'/'+sample_path.split('/')[-1] in validation_list):
                    np.save(save_dir+'/'+'validation'+'/'+str(cnt)+'.npy',data)
                else:
                    np.save(save_dir+'/'+'train'+'/'+str(cnt)+'.npy',data)
                cnt+=1
    
    print('process silence')
    sample_dir=data_dir+'/'+'_background_noise_'
    sample_list=os.listdir(sample_dir)
    sample_list.remove('README.md')
    for sample in sample_list:
        sample_path=sample_dir+'/'+sample
        signal, sr = librosa.load(sample_path,sr=16000)
        for i in range(200):
            idx=random.randint(0,len(signal)-16000)
            signal_tmp=signal[idx:idx+16000]
            mfccs=process_google_sample(signal_tmp)
            if(not(mfccs is None)):
                # continue
            # else:
                data={'feature':mfccs,'label':11}
                np.save((save_dir+'/'+'train'+'/'+str(cnt)+'.npy'),data)
                cnt+=1
    
    for sample in sample_list:
        sample_path=sample_dir+'/'+sample
        signal, sr = librosa.load(sample_path,sr=16000)
        for i in range(25):
            idx=random.randint(0,len(signal)-16000)
            signal_tmp=signal[idx:idx+16000]
            mfccs=process_google_sample(signal_tmp)
            if(not(mfccs is None)):
                # continue
            # else:
                data={'feature':mfccs,'label':11}
                np.save((save_dir+'/'+'test'+'/'+str(cnt)+'.npy'),data)
                cnt+=1
    
    for sample in sample_list:
        sample_path=sample_dir+'/'+sample
        signal, sr = librosa.load(sample_path,sr=16000)
        for i in range(25):
            idx=random.randint(0,len(signal)-16000)
            signal_tmp=signal[idx:idx+16000]
            mfccs=process_google_sample(signal_tmp)
            if(not(mfccs is None)):
                # continue
            # else:
                data={'feature':mfccs,'label':11}
                np.save((save_dir+'/'+'validation'+'/'+str(cnt)+'.npy'),data)
                cnt+=1
    
    print(f'finished. samples number: {cnt}')

prepare_google()