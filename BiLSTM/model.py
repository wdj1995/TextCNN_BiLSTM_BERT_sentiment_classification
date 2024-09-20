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


class BiLSTM_Length(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, max_length=10):
        """
        Define the layers of the module.

        vocab_size - vocabulary size
        embedding_dim - size of the dense word vectors
        hidden_dim - size of the hidden states
        output_dim - number of classes
        n_layers - number of multi-layer RNN
        bidirectional - boolean - use both directions of LSTM
        dropout - dropout probability
        pad_idx -  string representing the pad token
        """

        super().__init__()

        self.max_length = max_length #####################

        # 1. Feed the tweets in the embedding layer
        # padding_idx set to not learn the emedding for the <pad> token - irrelevant to determining sentiment
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # 2. LSTM layer
        # returns the output and a tuple of the final hidden state and final cell state
        self.encoder = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               dropout=dropout)

        # 3. Fully-connected layer
        # Final hidden state has both a forward and a backward component concatenated together
        # The size of the input to the nn.Linear layer is twice that of the hidden dimension size
        self.predictor = nn.Linear(hidden_dim * 2, output_dim)

        # Initialize dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        """
        The forward method is called when data is fed into the model.

        text - [tweet length, batch size]
        text_lengths - lengths of tweet
        """

        # Clamp everything to minimum length of 1, but keep the original variable to mask the output later
        # text_lengths_clamped = text_lengths.clamp(min=1, max=self.max_length) #####################

        # embedded = [sentence len, batch size, emb dim]
        embedded = self.dropout(self.embedding(text))

        # Pack the embeddings - cause RNN to only process non-padded elements
        # Speeds up computation
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths) ####################

        # output of encoder
        packed_output, (hidden, cell) = self.encoder(packed_embedded)

        # unpack sequence - transform packed sequence to a tensor
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sentence len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # Get the final layer forward and backward hidden states
        # concat the final forward and backward hidden layers and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.predictor(hidden)



class BiLSTM_No_Length(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout=0.5, hidden_size=128, num_layers=2):
        super(BiLSTM_No_Length, self).__init__()  # 初始化

        self.embeddings = nn.Embedding(vocab_size, embedding_size)  # 配置嵌入层，计算出词向量
        self.lstm = nn.LSTM(
            input_size = embedding_size,  # 输入大小为转化后的词向量
            hidden_size = hidden_size,  # 隐藏层大小
            num_layers = num_layers,  # 堆叠层数，有几层隐藏层就有几层
            dropout = dropout,  # 遗忘门参数
            bidirectional = True  # 双向LSTM
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            num_layers * hidden_size * 2,  # 因为双向所有要*2
            output_size
        )


    def forward(self, x):
        embedded = self.embeddings(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        feature = self.dropout(h_n)
        # 这里将所有隐藏层进行拼接来得出输出结果，没有使用模型的输出
        feature_map = torch.cat([feature[i, :, :] for i in range(feature.shape[0])], dim=-1)
        out = self.fc(feature_map)

        return out








