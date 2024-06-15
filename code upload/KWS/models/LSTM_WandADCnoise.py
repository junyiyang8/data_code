import os
import torch
import librosa
#import joblib
import time
import argparse
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from torch.nn import functional 

BITNUM_SIG=5
POINTNUMS=2**BITNUM_SIG

BITNUM_TANH=5
POINTNUMT=2**(BITNUM_TANH-1)

noise_std=2.6
print('WEIGHT_NOISE',noise_std)
max_conductance=50
print('max(conductance)',max_conductance)
# r_noise_std=3.5

def ramp_sigmoid_generation_noise():
    ##get V
    t= torch.arange(1/(POINTNUMS+2),1,1/(POINTNUMS+2))  #5 bits,33 points,32 delta_v
    # t= torch.arange(1/POINTNUMS,1,1/POINTNUMS)  #5 bits
    v=torch.log(t/(1-t))    #inverse function of sigmoid
    
    ##get Delta_V
    Delta_V=torch.zeros(len(v)-1).cuda()
    for i in range(len(v)-1):
        Delta_V[i]= v[i+1]-v[i]  
    #####get resulotion and cell_matrix 
    dummy = torch.round(Delta_V, decimals=3)
    resulotion = torch.min(dummy)
    numerator = torch.round(Delta_V, decimals=3)
    denominator_temp = torch.round(Delta_V, decimals=3)
    denominator = torch.min(denominator_temp)
    # print(denominator.size())
    cell_matrix=numerator/denominator
    # print(cell_matrix.size())
    # Delta_V_actvale=resulotion*cell_matrix       

    #addding noise
    g=((max_conductance/torch.max(cell_matrix))*cell_matrix).cuda()
    # print(g.size())   
    # Δg=torch.normal(0,5,(1,len(cell_matrix))).view(-1)
    # print('Δgsigmoid=',Δg)
    
    # print(g.size(), g_Δg_s.size())

    # Δg_r_s=torch.normal(0,r_noise_std,(1,(POINTNUMS))).view(-1).cuda() ###for test
    # Delta_V_actvale_dvt=(g+g_Δg_s+ Δg_r_s)*(torch.max(Delta_V)/150)   ###for test
    Delta_V_actvale_dvt=(g+g_Δg_s )*(torch.max(Delta_V)/150) ###for training
    # print('g_Δg_s=',g_Δg_s)
     #############get V_actvale
    V_actvale=torch.zeros(len(v))
    middle=int(len(Delta_V)/2)
    
    V_actvale[0]=-sum(Delta_V_actvale_dvt[:middle])
    for i in range(len(v)-1):
        V_actvale[i+1]= V_actvale[i]+Delta_V_actvale_dvt[i]
    return V_actvale

# SIGMOID_RAMP_NOISE = ramp_sigmoid_generation_noise()
# SIGMOID_RAMP_NOISE.requires_grad = False


def ramp_tanh_generation_noise():
      ##get V
    t= torch.arange(-(POINTNUMT)/(POINTNUMT+1),1,1/(POINTNUMT+1))  # 5bits 33 points(-16/17---16/17,1/17),32 delta_v  
    # t= torch.arange(-(POINTNUMT-1)/POINTNUMT,1,1/POINTNUMT)  # 5bits
    v=0.5*torch.log((1+t)/(1-t))     #inverse function of tanh
    # print(v)
    ##get Delta_V
    Delta_V=torch.zeros((len(v)-1)).cuda()

    for i in range(len(v)-1):
        Delta_V[i]= v[i+1]-v[i]  

    #####get resulotion and cell_matrix 
    dummy = torch.round(Delta_V, decimals=3)
    resulotion = torch.min(dummy)
    numerator = torch.round(Delta_V, decimals=3)
    denominator_temp = torch.round(Delta_V, decimals=3)
    denominator = torch.min(denominator_temp)
    cell_matrix=numerator/denominator
    # Delta_V_actvale=resulotion*cell_matrix        ##addding noise
    g=((max_conductance/torch.max(cell_matrix))*cell_matrix).cuda()
    # Δg=torch.normal(0,5,(1,len(cell_matrix))).view(-1)
    # print('Δgtanh=',g_Δg_t)


    # Δg_r_t=torch.normal(0,r_noise_std,(1,(POINTNUMS))).view(-1).cuda() ###for test
    # Delta_V_actvale_dvt=(g+g_Δg_s+ Δg_r_t)*(torch.max(Delta_V)/150)   ###for test
    Delta_V_actvale_dvt=(g+g_Δg_t)*(torch.max(Delta_V)/150) ###for training
    



    #############get V_actvale
    V_actvale=torch.zeros(len(v))
    middle=int(len(Delta_V)/2)
    V_actvale[0]=-sum(Delta_V_actvale_dvt[:middle])
    for i in range(len(v)-1):
        V_actvale[i+1]= V_actvale[i]+Delta_V_actvale_dvt[i]


    return V_actvale

# TANH_RAMP_NOISE = ramp_tanh_generation_noise()
# TANH_RAMP_NOISE.requires_grad = False

class ADCSigmoidNoise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        counter = torch.zeros(input.shape, device=input.device)
        for i in range(POINTNUMS-1):
            check = input >= G_SIGMOID_RAMP_NOISE[i]
            counter += check
            
        return counter*(1/(POINTNUMS+2))  ##counter*1/34
    
    @staticmethod
    def backward(ctx, grad_output):
        (input, ) = ctx.saved_tensors
        grad_input = grad_output * (torch.sigmoid(input) * (1 - torch.sigmoid(input)))

        return grad_input

def adc_sigmoid_noise():
    def inner(x):
        return ADCSigmoidNoise.apply(x)
    
    return inner

class ADCTanhNoise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        counter = torch.zeros(input.shape,device=input.device)
        for i in range(POINTNUMS-1):
            check = input >= G_TANH_RAMP_NOISE[i]
            counter += check

        return counter*(1/((POINTNUMS+2)/2))-1  ##counter*1/17-1
    
    @staticmethod
    def backward(ctx, grad_output):
        (input, ) = ctx.saved_tensors
        grad_input = grad_output * (1 - torch.tanh(input)*torch.tanh(input))

        return grad_input

def adc_tanh_noise():
    def inner(x):
        return ADCTanhNoise.apply(x)
    
    return inner  


class LSTMAdcNoise(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_mu = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U_mu = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias_mu = nn.Parameter(torch.Tensor(hidden_size * 4))

        self.W=torch.zeros(input_size, hidden_size * 4).cuda()
        self.U=torch.zeros(hidden_size, hidden_size * 4).cuda()
        self.bias=torch.zeros(hidden_size * 4).cuda()

        self.init_weights()
        self.dropout = nn.Dropout(p=0.01)   #aviod overfitting
        # self.sigmoid_act = adc_sigmoid_noise()
        # self.tanh_act = adc_tanh_noise()
        # global G_SIGMOID_RAMP_NOISE
        # G_SIGMOID_RAMP_NOISE= adc_sigmoid_noise()
        # global G_TANH_RAMP_NOISE
        # G_TANH_RAMP_NOISE= adc_tanh_noise()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                init_states=None,):
        """Assumes x is of shape (batch, sequence, feature)"""
        
        
        SIGMOID_RAMP_NOISE = ramp_sigmoid_generation_noise()
        SIGMOID_RAMP_NOISE.requires_grad = False
        global G_SIGMOID_RAMP_NOISE
        G_SIGMOID_RAMP_NOISE=SIGMOID_RAMP_NOISE

        TANH_RAMP_NOISE = ramp_tanh_generation_noise()
        TANH_RAMP_NOISE.requires_grad = False
        global G_TANH_RAMP_NOISE
        G_TANH_RAMP_NOISE=TANH_RAMP_NOISE

        sigmoid_act = adc_sigmoid_noise()
        tanh_act = adc_tanh_noise()

        # print('net.lstm.W_mu',self.W_mu.max(),self.W_mu.min())
        # print('net.lstm.U__mu',self.U_mu.max(),self.U_mu.min())
        # print('net.lstm.bais_mu',self.bias_mu.max(),self.bias_mu.min())

        

        LSTM_delg=torch.normal(0,noise_std,(73,128)).cuda() ###for training
        # LSTM_delg=torch.normal(0,r_noise_std,(73,128)).cuda()  ##for test
        # LSTM_delg=torch.zeros(73,128).cuda()
        # with open("result_drop001_noise10_bit5_v3_trynoise.txt", "a") as f:
        #     f.write(f'LSTM_delg[0]=={LSTM_delg[0]}\n')
        # print('LSTM_delg[0][0]',LSTM_delg[0][0])
        LSTM_Wandbias_max=2
        g_ratio_lstm=max_conductance/LSTM_Wandbias_max
        with torch.no_grad():
            a=torch.abs(self.W_mu)>=LSTM_Wandbias_max
            # print(type(a))
            b=LSTM_Wandbias_max*torch.sign(self.W_mu[a])
            self.W_mu[a]=b
            self.U_mu[torch.abs(self.U_mu)>=LSTM_Wandbias_max]=LSTM_Wandbias_max*torch.sign(self.U_mu[torch.abs(self.U_mu)>=LSTM_Wandbias_max])
            self.bias_mu[torch.abs(self.bias_mu)>=LSTM_Wandbias_max]=LSTM_Wandbias_max*torch.sign(self.bias_mu[torch.abs(self.bias_mu)>=LSTM_Wandbias_max])

      
        # self.W=(LSTM_delg[:40,:]/g_ratio_lstm+self.W_mu)
        # self.U=(LSTM_delg[40:72,:]/g_ratio_lstm+self.U_mu)
        # self.bias=(LSTM_delg[72:,:]/g_ratio_lstm+self.bias_mu)

        self.W=self.dropout(LSTM_delg[:40,:]/g_ratio_lstm+self.W_mu)   ##adding drop out
        self.U=self.dropout(LSTM_delg[40:72,:]/g_ratio_lstm+self.U_mu)  ##adding drop out
        self.bias=self.dropout(LSTM_delg[72:,:]/g_ratio_lstm+self.bias_mu)  #adding drop out

        # print('net.lstm.W_new',net.lstm.W.max(),net.lstm.W.min())
        # print('net.lstm.U_new',net.lstm.U.max(),net.lstm.U.min())
        # print('net.lstm.bais_new',net.lstm.bias.max(),net.lstm.bias.min())



        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            # gates = x_t @ self.W + h_t @ self.U + self.bias
            gates = x_t @ self.W + h_t @ self.U         ###no bias
            i_t, f_t, g_t, o_t = (
                sigmoid_act(gates[:, :HS]), # input
                # torch.sigmoid(gates[:, :HS]),
                sigmoid_act(gates[:, HS:HS*2]), # forget
                # torch.sigmoid(gates[:, HS:HS*2]),
                tanh_act(gates[:, HS*2:HS*3]),
                # torch.tanh(gates[:, HS*2:HS*3]),
                sigmoid_act(gates[:, HS*3:]), # output
                # torch.sigmoid(gates[:, HS*3:]),
            )
            c_t = f_t * c_t + i_t * g_t
            # h_t = o_t * tanh_act(c_t)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        # reshape from shape (sequence,feature ,batch ) to (batch, sequence, feature)   
        hidden_seq = torch.cat(hidden_seq, dim=0)      
        
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        return hidden_seq, (h_t, c_t)
##Custom Model


##AdcNoise Model
class LSTM_ClassificationAdcNoise(nn.Module):
    def __init__(self,input_size=40, hidden_size=32, num_class=12):
        super().__init__()
        # self.prelayer=nn.Linear(input_size,hidden_size)

        self.norm_layer=nn.LayerNorm(input_size,elementwise_affine=False)
        # self.norm_layer=nn.LayerNorm(input_size)

        self.dropout = nn.Dropout(p=0.4)   #aviod overfitting

        # self.lstm=LSTM(input_size=input_size, hidden_size=hidden_size)
        self.lstm=LSTMAdcNoise(input_size=input_size, hidden_size=hidden_size)

        self.linear_weight_mu=nn.Parameter(torch.randn(num_class, hidden_size)* 0.01)
        self.linear_bias_mu=nn.Parameter(torch.randn(num_class)*0.01)
        self.linear_weight=torch.zeros(num_class, hidden_size).cuda()
        self.linear_bias=torch.zeros(num_class).cuda()

        self.postlayer=nn.Linear(hidden_size,num_class)  

    def forward(self,x,Δg_s,Δg_t):
        # x=self.prelayer(x)

        # x=self.norm_layer(x)

        # self.dropout(x)
        global g_Δg_s
        g_Δg_s=Δg_s
        global g_Δg_t
        g_Δg_t=Δg_t
        output,(h,c)=self.lstm(x)    #output.shape = [batch_size,seq_length, , feature]; h,c=num_layers*num_directions，batch,hidden_size

        ####coustom FC layer
        # linear_delg=torch.normal(0,noise_std,(12,33)).cuda()
        # linear_Wandbias_max=4
        # g_ratio_linear=150/linear_Wandbias_max
        # with torch.no_grad():
        #     c=torch.abs(self.linear_weight_mu)>=linear_Wandbias_max
        #     # print(type(a))
        #     d=linear_Wandbias_max*torch.sign(self.linear_weight_mu[c])
        #     self.linear_weight_mu[c]=d
        #     # self.U_mu[torch.abs(self.U_mu)>=LSTM_Wandbias_max]=LSTM_Wandbias_max*torch.sign(self.U_mu[torch.abs(self.U_mu)>=LSTM_Wandbias_max])
        #     self.linear_bias_mu[torch.abs(self.linear_bias_mu)>=linear_Wandbias_max]=linear_Wandbias_max*torch.sign(self.linear_bias_mu[torch.abs(self.linear_bias_mu)>=linear_Wandbias_max])
        
        # # self.linear_weight=(linear_delg[:,:32]/g_ratio_linear+self.linear_weight_mu)
        # # self.linear_bias=(linear_delg[:,32:].view(-1)/g_ratio_linear+self.linear_bias_mu)

        # self.linear_weight=self.dropout(linear_delg[:,:32]/g_ratio_linear+self.linear_weight_mu) ##adding drop out
        # self.linear_bias=self.dropout(linear_delg[:,32:].view(-1)/g_ratio_linear+self.linear_bias_mu)  ##adding drop out

        # logits=functional.linear(output[:,-1,:],self.linear_weight, self.linear_bias)



      

        logits=self.postlayer(output[:,-1,:])  #logits.sahpe=128*1*num_class class=5, 
        logits=self.dropout(logits)
        return logits
