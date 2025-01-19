import torch 
from torch import nn
import numpy as np
#import matplotlib.pyplot as plt
from torch.autograd import Variable


def seq_max_pool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq = seq - (1 - mask) * 1e10
    return torch.max(seq, 1)

# def seq_min_pool(x):
#     """seq是[None, seq_len, s_size]的格式，
#     mask是[None, seq_len, 1]的格式，先除去mask部分，
#     然后再做maxpooling。
#     """
#     seq, mask = x
#     seq = seq + (1 - mask) * 1e10
#     return torch.min(seq, 1)

def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq , vec  = x
    vec = torch.unsqueeze(vec,1) #[16,1,128]
    
    vec = torch.zeros_like(seq[:, :, :1]) + vec #[16,457,128]
    return torch.cat([seq, vec], 2)

def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x #[16,457,128],[16,1]
    batch_idxs = torch.arange(0,seq.size(0)).cuda() #[16]

    batch_idxs = torch.unsqueeze(batch_idxs,1) #[16,1]

    idxs = torch.cat([batch_idxs, idxs], 1) #[16,2]

    res = []
    for i in range(idxs.size(0)):
        vec = seq[idxs[i][0],idxs[i][1],:] #提取头实体的开始索引对应的向量 [128]
        res.append(torch.unsqueeze(vec,0)) 
    
    res = torch.cat(res) #[16,128]
    return res


class s_model(nn.Module):
    def __init__(self,word_dict_length,word_emb_size,lstm_hidden_size):
        super(s_model,self).__init__()

        self.embeds = nn.Embedding(word_dict_length, word_emb_size).cuda()
        self.fc1_dropout = nn.Sequential(
            nn.Dropout(0.25).cuda(),  # drop 20% of the neuron 
        ).cuda()


        self.lstm1 = nn.LSTM(
            input_size = word_emb_size,
            hidden_size = int(word_emb_size/2),
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        ).cuda()


        self.lstm2 = nn.LSTM(
            input_size = word_emb_size,
            hidden_size = int(word_emb_size/2),
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        ).cuda()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size*2, #输入的深度
                out_channels=word_emb_size,#filter 的个数，输出的高度
                kernel_size = 3,#filter的长与宽
                stride=1,#每隔多少步跳一下
                padding=1,#周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ).cuda(),
            nn.ReLU().cuda(),
        ).cuda()
        self.fc_ps1 = nn.Sequential(
            nn.Linear(word_emb_size,1),
            nn.Sigmoid()
        ).cuda()

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size,1),
            nn.Sigmoid()
        ).cuda()

    def forward(self,t):
        mask = torch.gt(torch.unsqueeze(t,2),0).type(torch.cuda.FloatTensor) #(batch_size,sent_len,1)[16,457,1]
        mask.requires_grad = False
  
        outs = self.embeds(t) #[16,457,128]

        t = outs
        t = self.fc1_dropout(t) #[16,457,128]

        

        t = t.mul(mask) # (batch_size,sent_len,char_size) [16,457,128]  是点乘，要求ab的维度完全一致

        t, (h_n, c_n) = self.lstm1(t,None) #[16,457,128]
        t, (h_n, c_n) = self.lstm2(t,None) #[16,457,128]

        t_max,t_max_index = seq_max_pool([t,mask]) #[16,128]
  

        t_dim = list(t.size())[-1]
        h = seq_and_vec([t, t_max]) #[16,457,128*2]
  

        h = h.permute(0,2,1) #[16,256,457]
       
        h = self.conv1(h) #[16,128,457]
    
        h = h.permute(0,2,1) #[16,457,128]


        ps1 = self.fc_ps1(h) #[16,457,1]
        ps2 = self.fc_ps2(h) #[16,457,1]
        

        return [ps1.cuda(),ps2.cuda(),t.cuda(),t_max.cuda(),mask.cuda()]

class po_model(nn.Module):
    def __init__(self,word_dict_length,word_emb_size,lstm_hidden_size,num_classes):
        super(po_model,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size*4, #输入的深度
                out_channels=word_emb_size,#filter 的个数，输出的高度
                kernel_size = 3,#filter的长与宽
                stride=1,#每隔多少步跳一下
                padding=1,#周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ).cuda(),
            nn.ReLU().cuda(),
        ).cuda()

        self.fc_ps1 = nn.Sequential(
            nn.Linear(word_emb_size,num_classes+1).cuda(),
            #nn.Softmax(),
        ).cuda()

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size,num_classes+1).cuda(),
            #nn.Softmax(),
        ).cuda()
    
    def forward(self,t,t_max,k1,k2):

        k1 = seq_gather([t,k1]) #[16,128]

        k2 = seq_gather([t,k2]) #[16,128]

        k = torch.cat([k1,k2],1) # [16,256]
        h = seq_and_vec([t,t_max]) #[16,457,256]
        h = seq_and_vec([h,k]) #[16,457,256+256]
        h = h.permute(0,2,1) #[16,512,457]
        h = self.conv1(h) #[16,128,457]
        h = h.permute(0,2,1) #[16,457,128]

        po1 = self.fc_ps1(h) #[16,457,19]
        po2 = self.fc_ps2(h) #[16,457,19]

        return [po1.cuda(),po2.cuda()]






