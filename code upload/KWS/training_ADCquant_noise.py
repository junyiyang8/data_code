
import torch
import torch.nn as nn
import os
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from data_utils.datasets5b import get_google_dataloaders ##for N(0,5)
from models.LSTM_WandADCnoise import LSTM_ClassificationAdcNoise
from utils.utils import init_exp,test_on_dataloader

def write_to_file(BITNUM_SIG):
    with open("maxcondu50adc_drop001_bit5_newdataset_noiseN5_nobiaslstm.txt", "a") as f:
        f.write(f'BITNUM_SIG={BITNUM_SIG}\n')
       

batch_size=256
print('batch_size=',batch_size)
args=init_exp()
train_loader,test_loader,validation_loader=get_google_dataloaders(bs=batch_size)

# torch.save(net.state_dict(),os.path.join(args.work_dir,'net_para.pth'))

criterion=nn.CrossEntropyLoss(reduction='none')
task_log={'train_loss':[],'acc_train':[],'acc_test':[],'acc_val':[]}
cnt=0
best_acc_val=0
net=LSTM_ClassificationAdcNoise().cuda()
optimizer=torch.optim.Adam(net.parameters(),lr=0.0003)

BITNUM_SIG=5
POINTNUMS=2**BITNUM_SIG
BITNUM_TANH=5

noise_std=5

print('BITNUM_SIG=',BITNUM_SIG)
print('NLADC noise_std=',noise_std)

write_to_file(BITNUM_SIG)
with open("maxcondu50adc_smallLR_bit5_noiseN26_nobiaslstm.txt", "a") as f:
    f.write(f'BITNUM_SIG={BITNUM_SIG}\n')
    f.write(f'NLADC noise_std={noise_std}\n')
for epoch in range(500):
    

    train_loss=[]
    time_ckpt=time.time()
    net.train()


    for i, (sample,label) in enumerate(train_loader):

        Δg_s=torch.normal(0,noise_std,(1,(POINTNUMS))).view(-1).cuda()

        Δg_t=torch.normal(0,noise_std,(1,(POINTNUMS))).view(-1).cuda()

        sample,label=sample.cuda(),label.long().cuda()

        clf_rs=net(sample,Δg_s,Δg_t).cuda()

        loss=criterion(clf_rs,label)
        train_loss+=loss.tolist()
        loss=loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
 
    train_loss=sum(train_loss)/len(train_loss)

   

    net.eval()
    acc_train=test_on_dataloader(net,Δg_s,Δg_t,train_loader)
    acc_test=test_on_dataloader(net,Δg_s,Δg_t,test_loader)
    acc_val=test_on_dataloader(net,Δg_s,Δg_t,validation_loader)

    task_log['train_loss'].append(train_loss)
    task_log['acc_train'].append(acc_train)
    task_log['acc_test'].append(acc_test)
    task_log['acc_val'].append(acc_val)

    # print(f"epoch {epoch}: train_loss {train_loss:.4f} train acc {acc_train:.4f} test acc {acc_test:.4f}\
    # validation acc {acc_val:.4f} lr {optimizer.state_dict()['param_groups'][0]['lr']:.4f} time delta {time.time()-time_ckpt:.4f}")
    # print('\n')
    with open("maxcondu50adc_smallLR_bit5_noiseN26_nobiaslstm.txt", "a") as f:
        f.write(f"epoch {epoch}: train_loss {train_loss:.4f} train acc {acc_train:.4f} test acc {acc_test:.4f}\
        validation acc {acc_val:.4f} lr {optimizer.state_dict()['param_groups'][0]['lr']:.4f} time delta {time.time()-time_ckpt:.4f}")    
        f.write('\n')
    
    time_ckpt=time.time()
    joblib.dump(task_log,os.path.join(args.work_dir,'task_log.job'))

    if(acc_val>best_acc_val):
        cnt=0
        best_acc_val=acc_val
        torch.save(net.state_dict(),os.path.join(args.work_dir,'net_para.pth'))
    else:
        cnt+=1
        if(epoch>100 and cnt>args.early_thresh):
            break
    

    # lr_scheduler.step()

fig = plt.figure()
plt.plot(task_log['acc_train'],label ='acc_train')
plt.plot(task_log['acc_test'],label ='acc_test')
plt.plot(task_log['acc_val'],label ='acc_val')
plt.legend(loc='upper left')
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY")
plt.savefig(os.path.join(args.work_dir,'maxcondu50adc_smallLR_bit5_noiseN26_nobiaslstm.jpg'), dpi=300)

with open("maxcondu50adc_smallLR_bit5_noiseN26_nobiaslstm.txt", "a") as f:
    f.write(f'best_acc_val={best_acc_val}\n')

print('best_acc_val=',best_acc_val)