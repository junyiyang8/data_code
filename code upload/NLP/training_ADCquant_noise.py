import torch
import random
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from utils import gram_schmidt,compute_bpc
from data import get_dataloader
from model_Customadc_noise import PTB_LSTM

EPOCHS=20
DATA_PATH='dataset'
BATCH_SIZE=8
SLICE_LENGTH=128
NUM_WORKERS=24
INPUT_SIZE=128
HIDDEN_SIZE=2016
PROJECTION_SIZE=504
DROPOUT=0.25
CLASSES_NUM=50
LR=1e-3
STEP_SIZE=50
GAMMA=0.1
MAX_NORM=5.0
NORM_TYPE=2
SEED=0

print("BATCH_SIZE=",BATCH_SIZE)
print("SLICE_LENGTH=",SLICE_LENGTH)

noiseadc_std=int(5)
print('ADC_NOISE',noiseadc_std)

BITNUM=5
POINTNUMS=2**BITNUM
print('adc bit number=',BITNUM)


def set_random_seed(seed=0): #reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, Δg_s,Δg_t,data_source):
    model.eval()
    total_metric = 0.
    count = 0
    with torch.no_grad():
        for data, target in tqdm(data_source):
            data, target = data.cuda(), target.cuda()
            output, _ = model(data,Δg_s,Δg_t)
            total_metric += len(data) * compute_bpc(output.view(-1, output.size(-1)), target.view(-1)).item()
            count+=len(data)
    return total_metric / count

set_random_seed(SEED)
print('Loading data...')
embedding=torch.randn(CLASSES_NUM,INPUT_SIZE)
embedding=gram_schmidt(embedding)
torch.save(embedding, 'embedding.pt')
TRANSFORM = lambda x: F.embedding(x, embedding)
train_loader, val_loader, test_loader, final_test_loader=get_dataloader(data_path=DATA_PATH, batch_size=BATCH_SIZE, slice_length=SLICE_LENGTH, num_workers=NUM_WORKERS, transform=TRANSFORM)

print('Building model...')
model=PTB_LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, classes_num=CLASSES_NUM, projection_size=PROJECTION_SIZE, dropout=DROPOUT).cuda()
# model=PTB_LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, classes_num=CLASSES_NUM,  dropout=DROPOUT).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

model.load_state_dict(torch.load('best_model_Customadcdynamicnoise5bit5_8and128.pt'))
print('traning noise aware results')
print('Training model...')
best_metric = 1e6
for epoch in range(EPOCHS):
    time_ckpt= time.time()
    model.train()
    for data, target in tqdm(train_loader):
        
        Δg_s=torch.normal(0,noiseadc_std,(1,(POINTNUMS))).view(-1).cuda()
        Δg_t=torch.normal(0,noiseadc_std,(1,(POINTNUMS))).view(-1).cuda()


        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, _ = model(data,Δg_s,Δg_t)
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM, norm_type=NORM_TYPE)
        optimizer.step()

    train_time=time.time()-time_ckpt
    time_ckpt=time.time()

    scheduler.step()
    # train_metric= evaluate(model, train_loader)
    # eval_trainset_time=time.time()-time_ckpt
    # time_ckpt=time.time()

    val_metric = evaluate(model, Δg_s,Δg_t,val_loader)
    eval_valset_time=time.time()-time_ckpt
    time_ckpt=time.time()

    test_metric = evaluate(model,Δg_s,Δg_t, test_loader)
    eval_testset_time=time.time()-time_ckpt
    time_ckpt=time.time()
    print('Epoch: {:02d} | Train Metric: {:.6f} | Val Metric: {:.6f} | Test Metric: {:.6f} | Train Time: {:.6f} | Eval Trainset Time: {:.6f} | Eval Valset Time: {:.6f} | Eval Testset Time: {:.6f}'.format(epoch, val_metric, val_metric, test_metric, train_time, eval_valset_time, eval_valset_time, eval_testset_time))
    if(val_metric<best_metric):
        best_metric=val_metric
        torch.save(model.state_dict(), 'best_model_Customadcdynamicnoise8bit{}_8and128.pt'.format(BITNUM))
    
    if((epoch+1)%10==0):
        test_metric = evaluate(model,Δg_s,Δg_t, final_test_loader)
        print('Epoch {:02d} Test Metric: {:.6f}'.format(epoch, test_metric))

# torch.save(model.state_dict(), 'best_model_Customadc8and128.pt')
print('Testing model...')
model.load_state_dict(torch.load('best_model_Customadcdynamicnoise8bit{}_8and128.pt'.format(BITNUM)))
test_metric = evaluate(model, Δg_s,Δg_t,final_test_loader)
print('Test Metric: {:.6f}'.format(test_metric))