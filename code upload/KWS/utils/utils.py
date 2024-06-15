import torch
import random
import sys
import time
import os
import argparse
import numpy as np
class Tee:
    def __init__(args, fname, mode="a"):
        args.stdout = sys.stdout
        args.file = open(fname, mode)

    def write(args, message):
        args.stdout.write(message)
        args.file.write(message)
        args.flush()

    def flush(args):
        args.stdout.flush()
        args.file.flush()

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))

def get_args():
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument('--early_thresh', type=int, default=100)
    parser.add_argument('--num_class', type=int, default=12)
    # parser.add_argument('--slice_size', type=int, default=64)
    # parser.add_argument('--hidden_size', type=int, default=128)
    # parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=100)

    args = parser.parse_args()
    return args

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    print(s)

def test_on_dataloader(model,Δg_s,Δg_t,dataloader):
    clf_rs_list=[]
    label_list=[]
   
    for i, (sample,label) in enumerate(dataloader):
        sample,label=sample.cuda(),label.long().cuda() #yjy20230324

        clf_rs=model(sample,Δg_s,Δg_t).max(dim=-1)[1]  #[1]:only get index of max propability  model(sample,Δg_s,Δg_t).shape=bs*class_num (256*12) 12: 12个概率

        clf_rs_list+=clf_rs.tolist()
        label_list+=label.tolist()

    acc=(np.array(clf_rs_list)==np.array(label_list)).sum()/len(clf_rs_list)
    return acc

def init_exp():
    work_dir=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())
    work_dir='./lstm_work_dirs/'+work_dir
    os.makedirs(work_dir)
    sys.stdout = Tee(os.path.join(work_dir, 'out.txt'))
    sys.stderr = Tee(os.path.join(work_dir, 'err.txt'))
    args=get_args()
    args.work_dir=work_dir
    print_args(args)
    print_environ()
    set_random_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    return args


# import os
# import torch
# import librosa
# #import joblib
# import time
# import argparse
# import torch.nn as nn
# from torch.utils.data import Dataset,DataLoader
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import math

# class WaveDataset(Dataset):    #sample 100*128,label:1 number 0-29
#     # def __init__(self, num_class=6,slice_size=128, length=100, data_dir="speech/augmented_dataset/augmented_dataset"):
#     def __init__(self, num_class=6,slice_size=128, length=100, data_dir="./audio"):
#         self.slice_size=slice_size
#         self.length=length
#         self.sample_rate=self.slice_size*self.length
#         self.data_dir=data_dir
#         self.classes=os.listdir(data_dir)[:num_class]  #class is a list including 30 pathes
        
#         self.num_class=num_class
#         self.classes_map={self.classes[i]:i for i in range(self.num_class)}
#         self.sample_list=[]
#         self.label_list=[]
#         for i in self.classes:
#             data_dir_tmp=os.path.join(data_dir,i)   # D:/cityU/HKcollaboration/tmp/tmp/audio/i
#             sample_list_tmp=os.listdir(data_dir_tmp) #list all files under data_dir_tmp
#             sample_list_tmp=[os.path.join(data_dir_tmp,sample_list) for sample_list in sample_list_tmp]
#             sample_pth_list_tmp=[]
#             for sample_pth in sample_list_tmp:
#                 sample,sr=librosa.load(sample_pth,sr=self.sample_rate,duration=1)  #read adio file,1s
#                 if(len(sample)==self.length*self.slice_size):
#                     sample_pth_list_tmp.append(sample_pth)
#             self.sample_list+=sample_pth_list_tmp
#             self.label_list+=[self.classes_map[i] for _ in range(len(sample_pth_list_tmp))]
        
#     def __len__(self):
#         return len(self.label_list)

#     def __getitem__(self, idx):
#         sample_pth=self.sample_list[idx]
#         sample,sr=librosa.load(sample_pth,sr=self.sample_rate,duration=1)  #sample:12800 sr=sample Rate
        
#         sample=sample.reshape(self.length,self.slice_size) #sample:100*128
#         label=np.array(self.label_list[idx])
#         # print(np.shape(sample),np.shape(label))
#         return sample, label
# # def get_time(wave_pth):
# #     f = wave.open(wave_pth, 'r')
# #     time_count = f.getparams().nframes/f.getparams().framerate
# #     return time_count



# def get_dataloader(num_class=6,slice_size=128, length=100):     ##DataLoader将Dataset对象或自定义数据类的对象封装成一个迭代器,这个迭代器可以迭代输出Dataset的内容,#(([128, 100, 128]), [128]) 
#     # torch.manual_seed(1337)
#     all_set=WaveDataset(num_class=num_class,slice_size=slice_size,length=length)
#     train_size=int(0.7*len(all_set))
#     test_size=int(0.2*len(all_set))
#     val_size=len(all_set)-train_size-test_size
#     train_set,test_set,val_set=torch.utils.data.random_split(all_set,(train_size,test_size,val_size))

#     #train_loader=DataLoader(train_set,batch_size=128,num_workers=8,shuffle=True, pin_memory=True)
#     #test_loader=DataLoader(test_set,batch_size=128,num_workers=8,shuffle=True, pin_memory=True)
#     #val_loader=DataLoader(val_set,batch_size=128,num_workers=8,shuffle=True, pin_memory=True)
#     train_loader=DataLoader(train_set,batch_size=128,num_workers=0,shuffle=True)
#     test_loader=DataLoader(test_set,batch_size=128,num_workers=0,shuffle=True)
#     val_loader=DataLoader(val_set,batch_size=128,num_workers=0,shuffle=True)
#     return train_loader,test_loader,val_loader

# def run_train(early_thresh=20,num_class=6,slice_size=128,hidden_size=64,length=100):
#     #time_dir=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())
#     #time_dir+=f'-num_class-{num_class}-slice_size-{slice_size}-hidden_size-{hidden_size}-length-{length}'

#     time_dir = "./training_result/"

#     if not os.path.exists(time_dir):
#         os.mkdir(time_dir)
#     for inter in range(1):
#         train_loader,test_loader,val_loader=get_dataloader(num_class,slice_size,length)
#         print("Finished Data Processing", flush=True)

#         net=LSTMSpeechClassificationTrain(input_size=slice_size, hidden_size=hidden_size,num_class=num_class).cuda()
#         torch.save(net.state_dict(),os.path.join(time_dir,'net_para.pth'))
#         optimizer=torch.optim.Adam(net.parameters(),lr=0.01)
#         # scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch: 0.97**epoch)
#         criterion=nn.CrossEntropyLoss(reduction='none')
#         task_log={'train_loss':[],'acc_train':[],'acc_test':[],'acc_val':[]}

#         cnt=0
#         best_acc_val=0
#         for epoch in range(100):
            
#             net.train()
#             train_loss=[]

#             for i, (sample,label) in enumerate(train_loader):
#                 sample,label=sample.cuda(),label.long().cuda()  #yjy20230324
#                 #label = label.long()
#                 clf_rs=net(sample)   ##clf_rs.sahpe=128*1*30 ,class=30
#                 loss=criterion(clf_rs,label)
#                 train_loss+=loss.tolist()
#                 loss=loss.mean()   #?

#                 optimizer.zero_grad()
#                 loss.backward()  #number
#                 optimizer.step()
#             # scheduler.step()
#             train_loss=sum(train_loss)/len(train_loss)
#             net.eval()
#             acc_train=test_on_dataloader(net,train_loader)
#             acc_test=test_on_dataloader(net,test_loader)
#             acc_val=test_on_dataloader(net,val_loader)

#             task_log['train_loss'].append(train_loss)
#             task_log['acc_train'].append(acc_train)
#             task_log['acc_test'].append(acc_test)
#             task_log['acc_val'].append(acc_val)

#             print(f"epoch {epoch}:  train_loss {train_loss} train acc {acc_train} test acc {acc_test} val_loader acc {acc_val}  lr {optimizer.state_dict()['param_groups'][0]['lr']}", flush=True)
            
#             #joblib.dump(task_log,os.path.join(time_dir,'task_log.job'))

#             if(acc_val>best_acc_val):
#                 cnt=0
#                 best_acc_val=acc_val
#                 torch.save(net.state_dict(),os.path.join(time_dir,'net_para.pth'))
#             else:
#                 cnt+=1
#                 if(epoch>50 and cnt>early_thresh):
#                     break
#         print('best_acc_val=',best_acc_val)
#         fig = plt.figure()
#         plt.plot(task_log['acc_train'],label ='acc_train')
#         plt.plot(task_log['acc_test'],label ='acc_test')
#         plt.plot(task_log['acc_val'],label ='acc_val')
#         plt.legend(loc='upper left')
#         plt.xlabel("EPOCH")
#         plt.ylabel("ACCURACY")
#         plt.savefig("acc_adc_both%d.jpg"%inter, dpi=200)
#         # plt.show()

#         print('finish training\n \n')
#     return(acc_test)

# def get_args():
#     parser = argparse.ArgumentParser(description='DG')
#     parser.add_argument('--early_thresh', type=int, default=20)
#     parser.add_argument('--num_class', type=int, default=30)
#     parser.add_argument('--slice_size', type=int, default=128)
#     parser.add_argument('--hidden_size', type=int, default=64)
#     parser.add_argument('--length', type=int, default=100)
#     parser.add_argument('--gpu', type=int, default=0)

#     args = parser.parse_args()
#     return args

# if __name__ == "__main__":
#     args=get_args()
#     torch.cuda.set_device(args.gpu) #20230324
#     acc_test_ideal1=run_train(early_thresh=args.early_thresh,num_class=args.num_class,slice_size=args.slice_size,hidden_size=args.hidden_size, length=args.length)  #train and test (idel)
