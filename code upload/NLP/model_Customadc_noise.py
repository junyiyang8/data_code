import torch.nn as nn
import torch
import math
# from train_Customadc_noise import BITNUM
ADCBITNUMBER=5

BITNUM_SIG=ADCBITNUMBER
POINTNUMS=2**BITNUM_SIG
print('adc bit number=',BITNUM_SIG)
BITNUM_TANH=ADCBITNUMBER
POINTNUMT=2**(BITNUM_TANH-1)

noise_std=int(5)
print('WEIGHT_NOISE',noise_std)

def ramp_sigmoid_generation_noise():
    ##get V
    t= torch.arange(1/(POINTNUMS+2),0.9999999,1/(POINTNUMS+2))  #5 bits,33 points,32 delta_v
    v=torch.log(t/(1-t))    #inverse function of sigmoid
    
    ##get Delta_V
    Delta_V=torch.zeros(len(v)-1).cuda()
    # print(Delta_V.size())  
    for i in range(len(v)-1):
        Delta_V[i]= v[i+1]-v[i]  
    #####get resulotion and cell_matrix 
    dummy = torch.round(Delta_V, decimals=3)
    resulotion = torch.min(dummy)
    numerator = torch.round(Delta_V, decimals=3)
    denominator_temp = torch.round(Delta_V, decimals=3)
    denominator = torch.min(denominator_temp)
    cell_matrix=numerator/denominator
    

    #addding noise
    g=((150/torch.max(cell_matrix))*cell_matrix).cuda()
    Delta_V_actvale_dvt=(g+g_Δg_s )*(torch.max(Delta_V)/150) ###dynamic noise

     #############get V_actvale
    V_actvale=torch.zeros(len(v))
    middle=int(len(Delta_V)/2)
    
    V_actvale[0]=-sum(Delta_V_actvale_dvt[:middle])
    for i in range(len(v)-1):
        V_actvale[i+1]= V_actvale[i]+Delta_V_actvale_dvt[i]
    return V_actvale




def ramp_tanh_generation_noise():
      ##get V
    t= torch.arange(-(POINTNUMT)/(POINTNUMT+1),1,1/(POINTNUMT+1))  # 5bits 33 points(-16/17---16/17,1/17),32 delta_v  
    v=0.5*torch.log((1+t)/(1-t))     #inverse function of tanh
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

    

        ##addding noise
    g=((150/torch.max(cell_matrix))*cell_matrix).cuda()
    Delta_V_actvale_dvt=(g+g_Δg_t)*(torch.max(Delta_V)/150) ##dynamic noise
    

    #############get V_actvale
    V_actvale=torch.zeros(len(v))
    middle=int(len(Delta_V)/2)
    V_actvale[0]=-sum(Delta_V_actvale_dvt[:middle])
    for i in range(len(v)-1):
        V_actvale[i+1]= V_actvale[i]+Delta_V_actvale_dvt[i]

    return V_actvale

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
    def __init__(self, input_size, hidden_size,batch_first,proj_size=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(proj_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()
        self.dropout = nn.Dropout(p=0.01)   #aviod overfitting

        self.proj_layer = nn.Linear(hidden_size,proj_size)    
        self.proj_size = proj_size



        self.W_mu=torch.zeros(input_size, hidden_size * 4).cuda()   
        self.U_mu=torch.zeros(proj_size, hidden_size * 4).cuda()
        self.bias_mu=torch.zeros(hidden_size * 4).cuda()


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                init_states=None):
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



        

        LSTM_delg=torch.normal(0,noise_std,(633,8064)).cuda()
        LSTM_Wandbias_max=2
        g_ratio_lstm=150/LSTM_Wandbias_max
        with torch.no_grad():
            a=torch.abs(self.W)>=LSTM_Wandbias_max
            # print(type(a))
            b=LSTM_Wandbias_max*torch.sign(self.W[a])
            self.W[a]=b
            self.U[torch.abs(self.U)>=LSTM_Wandbias_max]=LSTM_Wandbias_max*torch.sign(self.U[torch.abs(self.U)>=LSTM_Wandbias_max])
            self.bias[torch.abs(self.bias)>=LSTM_Wandbias_max]=LSTM_Wandbias_max*torch.sign(self.bias[torch.abs(self.bias)>=LSTM_Wandbias_max])

        self.W_mu=self.dropout(LSTM_delg[:128,:]/g_ratio_lstm+self.W)   ##adding drop out
        self.U_mu=self.dropout(LSTM_delg[128:632,:]/g_ratio_lstm+self.U)  ##adding drop out
        self.bias_mu=self.dropout(LSTM_delg[632:,:]/g_ratio_lstm+self.bias)  #adding drop out        
               
        bs, seq_sz, _ = x.size()
        # print('x.size()=',x.size())
        hidden_seq = []
        if init_states is None:
            if self.proj_size > 0:
                h_t, c_t = (torch.zeros(bs, self.proj_size).to(x.device), 
                            torch.zeros(bs, self.hidden_size).to(x.device))
            else:
                h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                            torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            gates = x_t @ self.W_mu + h_t @ self.U_mu + self.bias_mu
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

            h_t = o_t * torch.tanh(c_t)

            h_t=self.proj_layer(h_t)  ##adding projection

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)

        

        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class PTB_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, projection_size, classes_num, dropout):
        super(PTB_LSTM, self).__init__() 
        self.lstm=LSTMAdcNoise(input_size, hidden_size, batch_first=True, proj_size=projection_size)

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(projection_size, classes_num)

    def forward(self, x,Δg_s,Δg_t):

        global g_Δg_s
        g_Δg_s=Δg_s
        global g_Δg_t
        g_Δg_t=Δg_t
        output, hidden = self.lstm(x)
        output = self.linear(output)
        return output,hidden




