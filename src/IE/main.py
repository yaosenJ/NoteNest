#! -*- coding:utf-8 -*-

import json
import numpy as np
from random import choice
from tqdm import tqdm
import model
import torch
#import data_prepare
import torch.utils.data as Data

import time
torch.backends.cudnn.benchmark = True


CHAR_SIZE = 128
SENT_LENGTH = 128  #4
HIDDEN_SIZE = 64
EPOCH_NUM = 500 # 100
#EPOCH_NUM=1
BATCH_SIZE = 16 #64

def get_now_time():
	a = time.time()  # 获取当前的时间戳，单位为秒（自1970年1月1日以来的秒数）
	return time.ctime(a) # 将时间戳格式化为可读的字符串时间表示

def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    #print("ML",ML)
    return [x + [0] * (ML - len(x)) for x in X]

def seq_padding_vec(X):
    L = [len(x) for x in X]
    ML = max(L)
    #print("ML",ML)
    return [x + [[1,0]] * (ML - len(x)) for x in X]

train_data = json.load(open(r'/root/kg-baseline-pytorch-master/kg-baseline-pytorch-master/coal/train_data.json',encoding='utf-8')) #157
dev_data = json.load(open(r'/root/kg-baseline-pytorch-master/kg-baseline-pytorch-master/coal/test_data.json',encoding='utf-8')) #30
id2predicate, predicate2id = json.load(open(r'/root/kg-baseline-pytorch-master/kg-baseline-pytorch-master/coal/relation_dict.json',encoding='utf-8'))
id2predicate = {int(i):j for i,j in id2predicate.items()}
id2char, char2id = json.load(open(r'/root/kg-baseline-pytorch-master/kg-baseline-pytorch-master/coal/all_chars_dict.json',encoding='utf-8'))
num_classes = len(id2predicate)

log_file = "/root/kg-baseline-pytorch-master/kg-baseline-pytorch-master/coal/training_log_batch_size_16_word_emb_size_128_epoch_500.txt"

class data_generator:
    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def pro_res(self):
        idxs = list(range(len(self.data)))
    	#print(idxs)
        np.random.shuffle(idxs)
        T, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], []
        for i in idxs:
            d = self.data[i]
            text = d['text']
            items = {}
            for sp in d['spo_list']:
                subjectid = text.find(sp[0])
                objectid = text.find(sp[2])
                if subjectid != -1 and objectid != -1:
                    key = (subjectid, subjectid+len(sp[0]))
                    if key not in items:
                        items[key] = []
                    items[key].append((objectid,objectid+len(sp[2]),predicate2id[sp[1]]))
            if items:
                T.append([char2id.get(c, 1) for c in text]) # 1是unk，0是padding
    			# s1, s2 = [[1,0]] * len(text), [[1,0]] * len(text)
                s1, s2 = [0] * len(text), [0] * len(text)
                for j in items:
    				# s1[j[0]] = [0,1]
    				# s2[j[1]-1] = [0,1]
                    s1[j[0]] = 1
                    s2[j[1]-1] = 1
    			#print(items.keys())
                k1, k2 = choice(list(items.keys()))
                o1, o2 = [0] * len(text), [0] * len(text) # 0是unk类（共49+1个类）
                for j in items[(k1, k2)]:
                    o1[j[0]] = j[2]
                    o2[j[1]-1] = j[2]
                S1.append(s1)
                S2.append(s2)
                K1.append([k1])
                K2.append([k2-1])
                O1.append(o1)
                O2.append(o2)


        T = np.array(seq_padding(T))
        S1 = np.array(seq_padding(S1))
        S2 = np.array(seq_padding(S2))
        O1 = np.array(seq_padding(O1))
        O2 = np.array(seq_padding(O2))
        K1, K2 = np.array(K1), np.array(K2)
        return [T, S1, S2, K1, K2, O1, O2]

# class data_generator:
#     def __init__(self, data, batch_size=BATCH_SIZE):
#         self.data = data
#         self.batch_size = batch_size
#         self.steps = len(self.data) // self.batch_size
#         if len(self.data) % self.batch_size != 0:
#             self.steps += 1

#     def __len__(self):
#         return self.steps

#     def pro_res(self):
#         idxs = list(range(len(self.data)))
#         np.random.shuffle(idxs)  # 打乱数据索引
#         T, S1, S2, K1, K2, O1, O2 = [], [], [], [], [], [], []

#         for i in idxs:
#             d = self.data[i]
#             text = d['text']
#             items = {}
#             for sp in d['spo_list']:
#                 subject_id = text.find(sp[0])
#                 object_id = text.find(sp[2])
#                 if subject_id != -1 and object_id != -1:
#                     key = (subject_id, subject_id + len(sp[0]))
#                     if key not in items:
#                         items[key] = []
#                     items[key].append((object_id, object_id + len(sp[2]), predicate2id[sp[1]]))

#             if items:  # 如果 items 非空，处理样本
#                 # 输入文本转为序列
#                 #T.append([char2id.get(c, 1) for c in text])  # 1 表示 UNK，0 表示 PAD

#                 # 初始化 S1 和 S2
#                 s1 = [0] * len(text)
#                 s2 = [0] * len(text)
#                 for start, end in items.keys():
#                     s1[start] = 1
#                     s2[end - 1] = 1

#                 # 遍历 items 中的所有 subject-object 对
#                 for start, end in items.keys():
#                     k1, k2 = start, end - 1
#                     K1.append([k1])
#                     K2.append([k2])

#                     o1 = [0] * len(text)  # 初始化 O1
#                     o2 = [0] * len(text)  # 初始化 O2
#                     for obj_start, obj_end, pred_id in items[(start, end)]:
#                         o1[obj_start] = pred_id
#                         o2[obj_end - 1] = pred_id

#                     T.append([char2id.get(c, 1) for c in text])  # 1 表示 UNK，0 表示 PAD
#                     S1.append(s1)
#                     S2.append(s2)
#                     O1.append(o1)
#                     O2.append(o2)

#         # 对所有序列进行 padding，使得每个样本的长度一致
#         T = np.array(seq_padding(T))       # [batch_size, max_seq_len]
#         S1 = np.array(seq_padding(S1))    # [batch_size, max_seq_len]
#         S2 = np.array(seq_padding(S2))    # [batch_size, max_seq_len]
#         O1 = np.array(seq_padding(O1))    # [batch_size, max_seq_len]
#         O2 = np.array(seq_padding(O2))    # [batch_size, max_seq_len]
#         K1 = np.array(K1)                 # [batch_size, 1]
#         K2 = np.array(K2)                 # [batch_size, 1]

#         return [T, S1, S2, K1, K2, O1, O2]

class myDataset(Data.Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """
    def __init__(self,_T,_S1,_S2,_K1,_K2,_O1,_O2):
        #xy = np.loadtxt('../dataSet/diabetes.csv.gz', delimiter=',', dtype=np.float32) # 使用numpy读取数据
        self.x_data = _T
        self.y1_data = _S1
        self.y2_data = _S2
        self.k1_data = _K1
        self.k2_data = _K2
        self.o1_data = _O1
        self.o2_data = _O2
        self.len = len(self.x_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y1_data[index],self.y2_data[index],self.k1_data[index],self.k2_data[index],self.o1_data[index],self.o2_data[index]

    def __len__(self):
        return self.len

def collate_fn(data):
    t = np.array([item[0] for item in data], np.int32)
    s1 = np.array([item[1] for item in data], np.int32)
    s2 = np.array([item[2] for item in data], np.int32)
    k1 = np.array([item[3] for item in data], np.int32)
    
    k2 = np.array([item[4] for item in data], np.int32)
    o1 = np.array([item[5] for item in data], np.int32)
    o2 = np.array([item[6] for item in data], np.int32)
    return {
      'T': torch.LongTensor(t), #[157,457]
      'S1': torch.FloatTensor(s1), #[157,457]
      'S2': torch.FloatTensor(s2), #[157,457]
	  'K1': torch.LongTensor(k1), #[157,1]
      'K2': torch.LongTensor(k2), #[157,1]
	  'O1': torch.LongTensor(o1), #[157,457]
      'O2': torch.LongTensor(o2), #[157,457]
    }

dg = data_generator(train_data)
T, S1, S2, K1, K2, O1, O2 = dg.pro_res()
# print("len",len(T))

torch_dataset = myDataset(T,S1,S2,K1,K2,O1,O2)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=8,
	collate_fn=collate_fn,      # subprocesses for loading data
)




# print("len",len(id2char))
s_m = model.s_model(len(char2id) + 2, CHAR_SIZE, HIDDEN_SIZE).cuda()
po_m = model.po_model(len(char2id) + 2, CHAR_SIZE, HIDDEN_SIZE, num_classes).cuda()
params = list(s_m.parameters())

params += list(po_m.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)


loss = torch.nn.CrossEntropyLoss(reduction='none').cuda() 
b_loss = torch.nn.BCELoss(reduction='none').cuda() 


def extract_items(text_in):
    R = []
    _s = [char2id.get(c, 1) for c in text_in]
    _s = np.array([_s])
    _k1, _k2,t , t_max,mask = s_m(torch.LongTensor(_s).cuda())
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _kk1s = []
    for i,_kk1 in enumerate(_k1):
        if _kk1 > 0.5:
            _subject = ''
            for j,_kk2 in enumerate(_k2[i:]):
                if _kk2 > 0.4:
                    _subject = text_in[i: i+j+1]
                    break
            if _subject:
                _k1, _k2 = torch.LongTensor([[i]]), torch.LongTensor([[i+j]]) #np.array([i]), np.array([i+j])
                _o1, _o2 = po_m(t.cuda(),t_max.cuda(),_k1.cuda(),_k2.cuda())
                _o1, _o2 = _o1.cpu().data.numpy(), _o2.cpu().data.numpy()

                _o1, _o2 = np.argmax(_o1[0], 1), np.argmax(_o2[0], 1) #[107],[107]

                for i,_oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for j,_oo2 in enumerate(_o2[i:]):
                            if _oo2 == _oo1:
                                _object = text_in[i: i+j+1]
                                _predicate = id2predicate[_oo1]
                                # print((_subject, _predicate, _object))
                                R.append((_subject, _predicate, _object))
                                break
        _kk1s.append(_kk1.data.cpu().numpy())
    _kk1s = np.array(_kk1s)
    return list(set(R))

def evaluate():
    A, B, C = 1e-10, 1e-10, 1e-10
    cnt = 0
    for d in tqdm(iter(dev_data)):
        R = set(extract_items(d['text']))
        T = set([tuple(i) for i in d['spo_list']])
        A += len(R & T)
        B += len(R)
        C += len(T)
        # if cnt % 1000 == 0:
        #     print('iter: %d f1: %.4f, precision: %.4f, recall: %.4f\n' % (cnt, 2 * A / (B + C), A / B, A / C))
        cnt += 1
    return 2 * A / (B + C), A / B, A / C


best_f1 = 0
best_epoch = 0

for i in range(EPOCH_NUM):
    for step, loader_res in tqdm(iter(enumerate(loader))):
        # print(get_now_time())
        t_s = loader_res["T"].cuda() #[16,457]
        k1 = loader_res["K1"].cuda() #[16,1]
        k2 = loader_res["K2"].cuda() #[16,1]
        s1 = loader_res["S1"].cuda() #[16,457]
        s2 = loader_res["S2"].cuda() #[16,457]
        o1 = loader_res["O1"].cuda() #[16,457]
        o2 = loader_res["O2"].cuda() #[16,457]

        ps_1,ps_2,t,t_max,mask = s_m(t_s)
        
        t,t_max,k1,k2 = t.cuda(),t_max.cuda(),k1.cuda(),k2.cuda() # 这些数据应用在预测关系和客体过程
        po_1,po_2 = po_m(t,t_max,k1,k2)
        
        ps_1 = ps_1.cuda() #[16,457,1]
        ps_2 = ps_2.cuda() #[16,457,1]
        po_1 = po_1.cuda() #[16,457,19]
        po_2 = po_2.cuda() #[16,457,19]
        
        s1 = torch.unsqueeze(s1,2) #[16,457,1]
        s2 = torch.unsqueeze(s2,2) #[16,457,1]


        s1_loss = b_loss(ps_1,s1) #[16,457,1]
        s1_loss = torch.sum(s1_loss.mul(mask))/torch.sum(mask) #mask:16,457,1
        s2_loss = b_loss(ps_2,s2) #[16,457,1]
        s2_loss = torch.sum(s2_loss.mul(mask))/torch.sum(mask)

        
        po_1 = po_1.permute(0,2,1) #[16,19,457]
        po_2 = po_2.permute(0,2,1)
        

        o1_loss = loss(po_1,o1) #[16,457]
        o1_loss = torch.sum(o1_loss.mul(mask[:,:,0])) / torch.sum(mask)
        o2_loss = loss(po_2,o2) #[16,457]
        o2_loss = torch.sum(o2_loss.mul(mask[:,:,0])) / torch.sum(mask)



        
        loss_sum = 2.5 * (s1_loss + s2_loss) + (o1_loss + o2_loss)

        # if step % 500 == 0:
        # 	torch.save(s_m, 'models_real/s_'+str(step)+"epoch_"+str(i)+'.pkl')
        # 	torch.save(po_m, 'models_real/po_'+str(step)+"epoch_"+str(i)+'.pkl')
        

        optimizer.zero_grad()

        loss_sum.backward()
        optimizer.step()


        
    # torch.save(s_m, r'/root/kg-baseline-pytorch-master/kg-baseline-pytorch-master/coal/models/s_'+str(i)+'.pkl')
    # torch.save(po_m, '/root/kg-baseline-pytorch-master/kg-baseline-pytorch-master/coal/models/po_'+str(i)+'.pkl')
    f1, precision, recall = evaluate()

    print("epoch:",i,"loss:",loss_sum.data)

    with open(log_file, "a") as f:
        f.write(f"epoch: {i}, loss: {loss_sum.data.item()}\n")
        
    if f1 >= best_f1:
        best_f1 = f1
        best_epoch = i
        torch.save(s_m, r'/root/kg-baseline-pytorch-master/kg-baseline-pytorch-master/coal/models/s_'+str(i)+'.pkl')
        torch.save(po_m, '/root/kg-baseline-pytorch-master/kg-baseline-pytorch-master/coal/models/po_'+str(i)+'.pkl')

    print('f1: %.4f, precision: %.4f, recall: %.4f, bestf1: %.4f, bestepoch: %d \n ' % (f1, precision, recall, best_f1, best_epoch))
    
