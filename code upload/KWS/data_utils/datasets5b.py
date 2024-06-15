from torch.utils.data import Dataset,DataLoader
import os
import numpy as np

class google_dataset(Dataset):
    def __init__(self, data_dir="./audio"):
        self.data_dir=data_dir
        self.data_list=os.listdir(data_dir)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data=np.load(self.data_dir+'/'+self.data_list[index],allow_pickle=True).tolist()
        return data['feature'],data['label']

def get_google_dataloaders(train_dir="./audio/train",test_dir="./audio/test",validation_dir="./audio/validation",bs=1024):
    
    train_data=google_dataset(train_dir)
    test_data=google_dataset(test_dir)
    validation_data=google_dataset(validation_dir)

    train_loader=DataLoader(train_data,batch_size=bs,num_workers=22,shuffle=True,pin_memory=True)
    test_loader=DataLoader(test_data,batch_size=bs,num_workers=22,shuffle=False,pin_memory=True)
    validation_loader=DataLoader(validation_data,batch_size=bs,num_workers=22,shuffle=False,pin_memory=True)

    return train_loader,test_loader,validation_loader

