import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os


#########################设置随机种子
torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)
os.environ['PYTHONHASHSEED'] = str(3407)  # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(3407)     # 为CPU设置随机种子
torch.cuda.manual_seed(3407)      # 为当前GPU设置随机种子（只用一块GPU）
# torch.cuda.manual_seed_all(3407)   # 为所有GPU设置随机种子（多块GPU）
torch.backends.cudnn.deterministic = True
#########################设置随机种子


class TextCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout = 0.5):
        super(TextCNN, self).__init__()

        Ci = 1
        Co = kernel_num

        self.embeddings = nn.Embedding(embed_num, embed_dim)
        # # 嵌入层加载预训练词向量
        # self.embed = self.embed.from_pretrained(vectors)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim),
                                               padding = (2, 0)) for f in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num, bias=True) # True代表自动进行softmax

    def forward(self, x):
        x = self.embeddings(x)  # (N, token_num, embed_dim)
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1) # (N, Co * len(kernel_sizes))
        x = self.dropout(x)  # (N, Co * len(kernel_sizes))
        logit = self.fc(x)  # (N, class_num)
        return logit


class TextCNNCSDN(nn.Module):
    def __init__(self,
                 class_num, # 最后输出的种类数
                 filter_sizes, # 卷积核的长也就是滑动窗口的长
                 filter_num,   # 卷积核的数量
                 vocabulary_size, # 词表的大小
                 embedding_dimension, # 词向量的维度
                 # vectors, # 词向量
                 dropout): # dropout率

        super(TextCNNCSDN, self).__init__() # 继承nn.Module

        chanel_num = 1  # 通道数，也就是一篇文章一个样本只相当于一个feature map

        self.embeddings = nn.Embedding(vocabulary_size, embedding_dimension) # 嵌入层
        # self.embedding = self.embedding.from_pretrained(torch.FloatTensor(vectors), freeze=False) #嵌入层加载预训练词向量

        # self.embed = nn.Embedding(vocabulary_size, embedding_dimension)  # embedding之后的shape: torch.Size([200, 8, 300])
        # self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        # self.embed.weight.data.copy_(torch.Tensor(vectors))
        # self.embed.weight.requires_grad = False

        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (fsz, embedding_dimension)) for fsz in filter_sizes])  # 卷积层
        self.dropout = nn.Dropout(dropout) # dropout
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num) #全连接层


    def forward(self, x):
        # x维度[句子长度,一个batch中所包含的样本数] 例:[3451,128]
        # x = self.static_embedding(x) # #经过嵌入层之后x的维度，[句子长度,一个batch中所包含的样本数,词向量维度] 例：[3451,128,300]
        # x = self.non_static_embedding(x) # #经过嵌入层之后x的维度，[句子长度,一个batch中所包含的样本数,词向量维度] 例：[3451,128,300]
        x = self.embeddings(x)

        x = x.permute(1,0,2) # permute函数将样本数和句子长度换一下位置，[一个batch中所包含的样本数,句子长度,词向量维度] 例：[128,3451,300]
        x = x.unsqueeze(1) # # conv2d需要输入的是一个四维数据，所以新增一维feature map数 unsqueeze(1)表示在第一维处新增一维，[一个batch中所包含的样本数,一个样本中的feature map数，句子长度,词向量维度] 例：[128,1,3451,300]
        x = [conv(x) for conv in self.convs] # 与卷积核进行卷积，输出是[一个batch中所包含的样本数,卷积核数，句子长度-卷积核size+1,1]维数据,因为有[3,4,5]三张size类型的卷积核所以用列表表达式 例：[[128,16,3459,1],[128,16,3458,1],[128,16,3457,1]]
        x = [sub_x.squeeze(3) for sub_x in x]#squeeze(3)判断第三维是否是1，如果是则压缩，如不是则保持原样 例：[[128,16,3459],[128,16,3458],[128,16,3457]]
        x = [F.relu(sub_x) for sub_x in x] # ReLU激活函数激活，不改变x维度
        x = [F.max_pool1d(sub_x,sub_x.size(2)) for sub_x in x] # 池化层，根据之前说的原理，max_pool1d要取出每一个滑动窗口生成的矩阵的最大值，因此在第二维上取最大值 例：[[128,16,1],[128,16,1],[128,16,1]]
        x = [sub_x.squeeze(2) for sub_x in x] # 判断第二维是否为1，若是则压缩 例：[[128,16],[128,16],[128,16]]
        x = torch.cat(x, 1) # 进行拼接，例：[128,48]
        x = self.dropout(x) # 去除掉一些神经元防止过拟合，注意dropout之后x的维度依旧是[128,48]，并不是说我dropout的概率是0.5，去除了一半的神经元维度就变成了[128,24]，而是把x中的一些神经元的数据根据概率全部变成了0，维度依旧是[128,48]
        logits = self.fc(x) # 全接连层 例：输入x是[128,48] 输出logits是[128,10]
        return logits




