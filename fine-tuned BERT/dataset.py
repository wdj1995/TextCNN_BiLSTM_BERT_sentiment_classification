import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
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

class SSTDataset(Dataset):
    """
    Stanford Sentiment Treebank V1.0
    Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
    Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
    Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
    """

    def __init__(self, filename, maxlen, tokenizer):
        # Store the contents of the file in pandas dataframe.
        self.df = pd.read_csv(filename, delimiter=",")
        # self.df = pd.read_csv(filename, delimiter="\t")
        # Initialize tokenizer for the desired transformer model.
        self.tokenizer = tokenizer
        # Maximum length of tokens list to keep all the sequences of fixed size.
        self.maxlen = maxlen

    def __len__(self):
        # Return length of dataframe.
        return len(self.df)

    def __getitem__(self, index):
        # Select sentence and label at specified index from data frame.
        sentence = self.df.loc[index, "sentence"]
        label = self.df.loc[index, "label"]
        # nn. CrossEntropyLoss需要整数标签，不需要onehot
        label = torch.tensor(label)
        # label = F.one_hot(label, num_classes=6) # 转换为one-hot编码
        # print("dataset one hot: ", label)

        # label = np.argmax(label) #使用CrossEntropyLoss时把one-hot编码修改为类别索引

        # Preprocess text to be suitable for transformer
        tokens = self.tokenizer.tokenize(sentence) # 进行了wordpiece分词
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        if len(tokens) < self.maxlen:
            tokens = tokens + ["[PAD]" for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[: self.maxlen - 1] + ["[SEP]"]

        # Obtain indices of tokens and convert them to tensor.
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens)) # 将数据从字符串转换为数字序列

        # Obtain attention mask i.e. a tensor containing 1s for no padded tokens and 0s for padded ones.
        attention_mask = (input_ids != 0).long()

        # Return input IDs, attention mask, and label.
        return input_ids, attention_mask, label
