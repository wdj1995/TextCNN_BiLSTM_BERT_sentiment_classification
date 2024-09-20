import torch.nn as nn
import torch
from transformers import (
    BertPreTrainedModel,
    BertModel,
    AlbertPreTrainedModel,
    AlbertModel,
    DistilBertPreTrainedModel,
    DistilBertModel,
)
# import analyzer
import random
import os
import numpy as np


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

class BertForSentimentClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # BERT.
        self.bert = BertModel(config)

        # for param in self.bert.parameters():
        # # print(name, param.shape)
        #     param.requires_grad = True

        # 冻结bert的1-6层参数
        # analyzer.Analyzer.freeze_unfreeze_layers(self.bert.encoder, (0, 5), unfreeze=False)

        # Classification layer, which takes [CLS] representation and outputs logits.
        # self.cls_layer = nn.Linear(config.hidden_size, 1) # 要实现六分类，修改输出参数为6
        """
        PyTorch的nn.Linear模块在进行线性变换后会自动应用softmax函数，从而得到对应类别的概率分布。
        这一设计使得神经网络的搭建和训练更加便捷，无需手动编写softmax函数。
        当我们使用nn.Linear时，可以直接将输出作为损失函数的输入，从而完成分类任务的训练。
        """
        self.cls_layer = nn.Linear(config.hidden_size, 6)
        # 要实现六分类，修改输出参数为6


    def forward(self, input_ids, attention_mask):
        """
        Inputs:
                -input_ids : Tensor of shape [B, T] containing token ids of sequences
                -attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
                (where B is the batch size and T is the input length)
        """
        # with torch.no_grad(): ## no_grad下参数不会迭代 ##

            # Feed input to BERT and obtain outputs.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Obtain representations of [CLS] heads.
        cls_reps = outputs.last_hidden_state[:, 0]

        # Put these representations to classification layer to obtain logits.
        logits = self.cls_layer(cls_reps)
        # Return logits.
        return logits



class AlbertForSentimentClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # ALBERT.
        self.albert = AlbertModel(config)
        self.cls_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Inputs:
                -input_ids : Tensor of shape [B, T] containing token ids of sequences
                -attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
                (where B is the batch size and T is the input length)
        """
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        cls_reps = outputs.last_hidden_state[:, 0]
        logits = self.cls_layer(cls_reps)
        return logits


class DistilBertForSentimentClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # DistilBERT.
        self.distilbert = DistilBertModel(config)
        self.cls_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Inputs:
                -input_ids : Tensor of shape [B, T] containing token ids of sequences
                -attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
                (where B is the batch size and T is the input length)
        """
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_reps = outputs.last_hidden_state[:, 0]
        logits = self.cls_layer(cls_reps)
        return logits


