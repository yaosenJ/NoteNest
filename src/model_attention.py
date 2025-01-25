# import torch
# from torch import nn
# import numpy as np




# class Attention(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Attention, self).__init__()
#         self.query_layer = nn.Linear(input_dim, output_dim)
#         self.key_layer = nn.Linear(input_dim, output_dim)
#         self.value_layer = nn.Linear(input_dim, output_dim)

#     def forward(self, seq, context):
#         query = self.query_layer(context)  # (batch_size, output_dim) [16,128]
#         key = self.key_layer(seq.reshape(-1, seq.size(-1))).reshape(seq.size(0), seq.size(1), -1) #[16,457,128]
#         value = self.value_layer(seq.reshape(-1, seq.size(-1))).reshape(seq.size(0), seq.size(1), -1)#[16,457,128]

#         attention_scores = torch.matmul(key, query.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)[16,457]
#         attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)

#         weighted_sum = torch.matmul(attention_weights.unsqueeze(1), value).squeeze(1)  # (batch_size, output_dim) #[16,128]
#         return weighted_sum, attention_weights

# def seq_max_pool(x):
#     seq, mask = x
#     seq = seq - (1 - mask) * 1e10
#     return torch.max(seq, 1)


# def seq_and_vec(x):
#     seq, vec = x
#     vec = torch.unsqueeze(vec, 1)#[16,1,128]
#     vec = torch.zeros_like(seq[:, :, :1]) + vec #[16,457,128]
#     return torch.cat([seq, vec], 2)


# def seq_gather(x):
#     seq, idxs = x
#     batch_idxs = torch.arange(0, seq.size(0)).cuda()
#     batch_idxs = torch.unsqueeze(batch_idxs, 1)
#     idxs = torch.cat([batch_idxs, idxs], 1)

#     res = []
#     for i in range(idxs.size(0)):
#         vec = seq[idxs[i][0], idxs[i][1], :]
#         res.append(torch.unsqueeze(vec, 0))

#     res = torch.cat(res)
#     return res


# class s_model(nn.Module):
#     def __init__(self, word_dict_length, word_emb_size, lstm_hidden_size):
#         super(s_model, self).__init__()
#         self.embeds = nn.Embedding(word_dict_length, word_emb_size).cuda()
#         self.fc1_dropout = nn.Sequential(nn.Dropout(0.25).cuda()).cuda()

#         self.lstm1 = nn.LSTM(
#             input_size=word_emb_size,
#             hidden_size=int(word_emb_size / 2),
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True
#         ).cuda()

#         self.lstm2 = nn.LSTM(
#             input_size=word_emb_size,
#             hidden_size=int(word_emb_size / 2),
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True
#         ).cuda()

#         self.attention = Attention(input_dim=word_emb_size, output_dim=word_emb_size).cuda()

#         self.conv1 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=word_emb_size * 2,
#                 out_channels=word_emb_size,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ).cuda(),
#             nn.ReLU().cuda(),
#         ).cuda()

#         self.fc_ps1 = nn.Linear(word_emb_size, 1).cuda()
#         self.fc_ps2 = nn.Linear(word_emb_size, 1).cuda()

#     def forward(self, t):
#         mask = torch.gt(torch.unsqueeze(t, 2), 0).type(torch.cuda.FloatTensor)#[16,457,1]
#         mask.requires_grad = False

#         outs = self.embeds(t)#[16,457,128]
#         t = self.fc1_dropout(outs)#[16,457,128]
#         t = t.mul(mask)#[16,457,128]

#         t, _ = self.lstm1(t, None)#[16,457,128]
#         t, _ = self.lstm2(t, None)#[16,457,128]

#         t_max, _ = seq_max_pool([t, mask]) #[16,128]
#         context, attention_weights = self.attention(t, t_max) #[16,128]

#         h = seq_and_vec([t, context]) #[16,457,256]
#         h = h.permute(0, 2, 1)#[16,256,457]
#         h = self.conv1(h) #[16,128,457]
#         h = h.permute(0, 2, 1)#[16,457,128]

#         ps1 = self.fc_ps1(h)#[16,457,1]
#         ps2 = self.fc_ps2(h)

#         return [ps1.cuda(), ps2.cuda(), t.cuda(), t_max.cuda(), mask.cuda()]


# class po_model(nn.Module):
#     def __init__(self, word_dict_length, word_emb_size, lstm_hidden_size, num_classes):
#         super(po_model, self).__init__()

#         self.attention = Attention(input_dim=word_emb_size * 2, output_dim=word_emb_size).cuda()

#         self.conv1 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=word_emb_size * 5,
#                 out_channels=word_emb_size,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ).cuda(),
#             nn.ReLU().cuda(),
#         ).cuda()

#         self.fc_ps1 = nn.Linear(word_emb_size, num_classes + 1).cuda()
#         self.fc_ps2 = nn.Linear(word_emb_size, num_classes + 1).cuda()

#     def forward(self, t, t_max, k1, k2):
#         k1 = seq_gather([t, k1]) #[16,128]
#         k2 = seq_gather([t, k2]) #[16,128]
#         k = torch.cat([k1, k2], 1) #[16,256]

#         h = seq_and_vec([t, t_max]) #[16,457,256]
#         h = seq_and_vec([h, k])#[16,457,256+256]

#         context, attention_weights = self.attention(h, k)

#         h = seq_and_vec([h, context])

#         h = h.permute(0, 2, 1)
#         h = self.conv1(h)
#         h = h.permute(0, 2, 1)

#         po1 = self.fc_ps1(h)
#         po2 = self.fc_ps2(h)

#         return [po1.cuda(), po2.cuda()]

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(input_dim, output_dim)
        self.key_layer = nn.Linear(input_dim, output_dim)
        self.value_layer = nn.Linear(input_dim, output_dim)

    def forward(self, seq, context):
        query = self.query_layer(context)  # (batch_size, output_dim) # context[16,128] query[16,128]
        key = self.key_layer(seq)  # (batch_size, seq_len, output_dim) # seq[16,457,128] key[16,457,128]
        value = self.value_layer(seq)  # (batch_size, seq_len, output_dim) # seq[16,457,128] value[16,457,128]

        attention_scores = torch.matmul(key, query.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)[16,457]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)

        weighted_sum = torch.matmul(attention_weights.unsqueeze(1), value).squeeze(1)  # (batch_size, output_dim) (16,128)
        return weighted_sum, attention_weights

def seq_max_pool(x):
    seq, mask = x
    seq = seq - (1 - mask) * 1e10
    return torch.max(seq, 1)

def seq_mean_pool(x):
    seq, mask = x
    # 计算序列的加权和，忽略 mask 中的无效部分
    sum_seq = torch.sum(seq * mask, dim=1)
    # 计算有效部分的长度
    mask_sum = torch.sum(mask, dim=1)
    # 避免除零错误
    mask_sum = torch.clamp(mask_sum, min=1e-10)
    # 返回均值
    return sum_seq / mask_sum


def seq_and_vec(x):
    seq, vec = x
    vec = torch.unsqueeze(vec, 1)  # (batch_size, 1, vec_dim)
    vec = torch.zeros_like(seq[:, :, :1]) + vec  # Broadcast vec to seq dimensions
    return torch.cat([seq, vec], 2)


def seq_gather(x):
    seq, idxs = x
    batch_idxs = torch.arange(0, seq.size(0)).cuda()
    batch_idxs = torch.unsqueeze(batch_idxs, 1)
    idxs = torch.cat([batch_idxs, idxs], 1)

    res = []
    for i in range(idxs.size(0)):
        vec = seq[idxs[i][0], idxs[i][1], :]
        res.append(torch.unsqueeze(vec, 0))

    res = torch.cat(res)
    return res


class s_model(nn.Module):
    def __init__(self, word_dict_length, word_emb_size, lstm_hidden_size):
        super(s_model, self).__init__()
        self.embeds = nn.Embedding(word_dict_length, word_emb_size).cuda()
        self.fc1_dropout = nn.Sequential(nn.Dropout(0.25).cuda())

        # self.lstm1 = nn.LSTM(
        #     input_size=word_emb_size,
        #     hidden_size=int(word_emb_size / 2),
        #     num_layers=1,
        #     batch_first=True,
        #     bidirectional=True
        # ).cuda()

        # self.lstm2 = nn.LSTM(
        #     input_size=word_emb_size,
        #     hidden_size=int(word_emb_size / 2),
        #     num_layers=1,
        #     batch_first=True,
        #     bidirectional=True
        # ).cuda()

        self.gru1 = nn.GRU(
        input_size=word_emb_size,
        hidden_size=int(word_emb_size / 2),
        num_layers=1,
        batch_first=True,
        bidirectional=True
        ).cuda()

        self.gru2 = nn.GRU(
            input_size=word_emb_size,
            hidden_size=int(word_emb_size / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True
        ).cuda()


        self.attention = Attention(input_dim=word_emb_size, output_dim=word_emb_size).cuda()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size * 2,
                out_channels=word_emb_size,
                kernel_size=3,
                stride=1,
                padding=1,
            ).cuda(),
            nn.ReLU().cuda(),
        )

        # self.fc_ps1 = nn.Linear(word_emb_size, 1).cuda()
        # self.fc_ps2 = nn.Linear(word_emb_size, 1).cuda()

        self.fc_ps1 = nn.Sequential(
            nn.Linear(word_emb_size,1),
            nn.Sigmoid()
        ).cuda()

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size,1),
            nn.Sigmoid()
        ).cuda()

    def forward(self, t):
        mask = torch.gt(torch.unsqueeze(t, 2), 0).type(torch.cuda.FloatTensor)  # (batch_size, seq_len, 1)
        mask.requires_grad = False

        outs = self.embeds(t)  # (batch_size, seq_len, embedding_dim)
        t = self.fc1_dropout(outs)  # (batch_size, seq_len, embedding_dim)
        t = t.mul(mask)  # Apply mask

        # t, _ = self.lstm1(t, None)  # (batch_size, seq_len, lstm_hidden_size*2)
        # t, _ = self.lstm2(t, None)

        t, _ = self.gru1(t, None)  # (batch_size, seq_len, gru_hidden_size*2)
        t, _ = self.gru2(t, None)


        # t_max, _ = seq_max_pool([t, mask])  # (batch_size, lstm_hidden_size*2)
        t_max = seq_mean_pool([t,mask])
        context, attention_weights = self.attention(t, t_max)  # (batch_size, lstm_hidden_size*2)

        h = seq_and_vec([t, context])  # (batch_size, seq_len, lstm_hidden_size*4)
        #h = seq_and_vec([h, t_max])
        h = h.permute(0, 2, 1)  # (batch_size, lstm_hidden_size*4, seq_len)
        h = self.conv1(h)  # (batch_size, lstm_hidden_size*2, seq_len)
        h = h.permute(0, 2, 1)  # (batch_size, seq_len, lstm_hidden_size*2)

        ps1 = self.fc_ps1(h)  # (batch_size, seq_len, 1)
        ps2 = self.fc_ps2(h)

        return [ps1.cuda(), ps2.cuda(), t.cuda(), t_max.cuda(), mask.cuda()]


class po_model(nn.Module):
    def __init__(self, word_dict_length, word_emb_size, lstm_hidden_size, num_classes):
        super(po_model, self).__init__()

        self.attention = Attention(input_dim=word_emb_size, output_dim=word_emb_size).cuda()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size * 5,
                out_channels=word_emb_size,
                kernel_size=3,
                stride=1,
                padding=1,
            ).cuda(),
            nn.ReLU().cuda(),
        )

        self.fc_ps1 = nn.Linear(word_emb_size, num_classes + 1).cuda()
        self.fc_ps2 = nn.Linear(word_emb_size, num_classes + 1).cuda()

    def forward(self, t, t_max, k1, k2):
        k1 = seq_gather([t, k1])  
        k2 = seq_gather([t, k2])  
        k = torch.cat([k1, k2], 1)  

        h = seq_and_vec([t, t_max])  
        context, attention_weights = self.attention(t, t_max)

        h = seq_and_vec([h, context])
        h = seq_and_vec([h, k])  
        h = h.permute(0, 2, 1)  
        h = self.conv1(h)  
        h = h.permute(0, 2, 1)  

        po1 = self.fc_ps1(h) 
        po2 = self.fc_ps2(h)

        return [po1.cuda(), po2.cuda()]

