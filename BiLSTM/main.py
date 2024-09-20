import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as data
import torchtext.datasets as datasets
import jieba
import argparse
import random
from torchtext import data
from model import BiLSTM_No_Length
# from model import TextCNNCSDN
from operation import *
import torchtext
from torchtext.vocab import Vectors


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


# Setting device on GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)


# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

def parse_arguments():
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument('-lr', type=float, default = 0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default = 200, help='number of epochs for train [default: 10]')
    parser.add_argument('-batch-size', type=int, default = 128, help='batch size for training [default: 128]')
    parser.add_argument('-log-interval',  type=int, default = 100,   help='how many steps to wait before logging training status [default: 100]')
    parser.add_argument('-test-interval', type=int, default = 200, help='how many steps to wait before testing [default: 200]')
    parser.add_argument('-save-interval', type=int, default = 1000, help='how many steps to wait before saving [default: 1000]')
    parser.add_argument('-save-dir', type=str, default='snapshot_English', help='directory to save the snapshot')
    # model
    parser.add_argument('-dropout', type=float, default = 0.5, help='dropout probability [default: 0.5]')
    # parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-embed-dim', type=int, default = 100, help='number of embedding dimension [default: 128]')
    # parser.add_argument('-kernel-num', type=int, default=10, help='number of kernels')
    parser.add_argument('-kernel-num', type=int, default = 16, help='number of kernels')
    parser.add_argument('-kernel-sizes', type=str, default = '3,4,5', help='comma-separated kernel size to use for convolution')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-train', action='store_true', default=False, help='train a new model')
    parser.add_argument('-test', action='store_true', default=False, help='test on testset, combined with -snapshot to load model')
    parser.add_argument('-predict', action='store_true', default=False, help='predict label of console input')
    args = parser.parse_args()

    return args


################################################################# Dataset #################################################################
################################################################# Dataset #################################################################
############# 分词和去停用词
stopwords_path = './stopwords/baidu_stopwords.txt'
# stopwords = open(stopwords_path).read().split('\n')
stopwords = [line.strip() for line in open(stopwords_path, encoding='UTF-8').readlines()]
##### 英文分词
import spacy
spacy_en = spacy.load('en_core_web_md')
def tokenize_EN(text):
    return [word.text for word in spacy_en.tokenizer(text)]
##### 英文分词

# def tokenize_CN(text):  # jieba是中文分词函数，进行英文文本分类时是否要进行英文分词？
#     # return [word for word in jieba.cut(text) if word.strip()] # jieba.cut(txt)精确模式进行分词
#     return [word for word in jieba.lcut(text) if word not in stopwords] # 使用分词表删除停用词
# 去停用词
def get_stop_words():
    file_object = open(stopwords_path, encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words
############# 分词和去停用词

args = parse_arguments()

# 数据集构建
stop_words = get_stop_words()  # 加载停用词表
# text_field = data.Field(lower=True, tokenize = tokenize_CN)
# text_field = data.Field(lower = True, tokenize = tokenize_CN, stop_words=stop_words)
# text_field = data.Field(lower = True, tokenize = tokenize_EN, stop_words = stop_words, include_lengths=True)
text_field = data.Field(lower = True, tokenize = tokenize_EN, stop_words = stop_words)
# text_field = data.Field(lower = True, tokenize = tokenize_EN, stop_words = stop_words)
# label_field = data.Field(sequential = False) # sequential是否把数据表示成序列,如果是False,不能使用分词,默认值: True.
label_field = data.Field(sequential=False, dtype=torch.float32, use_vocab=False, unk_token=None) # sequential是否把数据表示成序列,如果是False,不能使用分词,默认值: True.

# fields = [('text', text_field), ('label', label_field)]
fields = [('sentence', text_field), ('label', label_field)]

# train_dataset, val_dataset = data.TabularDataset.splits(
#     path = './data/English_2classes/', format = 'csv', skip_header = True,
#     train = 'train_SST2.csv', validation= 'val_SST2.csv',
#     fields = fields
# )

train_dataset, val_dataset, test_dataset = data.TabularDataset.splits(
    path = './data/English_6classes/', format = 'csv', skip_header = True,
    train = 'emotion_dataset_6classes_train.csv', validation = 'emotion_dataset_6classes_val.csv',
    test = 'emotion_dataset_6classes_test.csv',
    fields = fields
)
"""
dataset = data.TabularDataset(
    path = './data/English_6classes/emotion_dataset_6classes.csv',
    format = 'csv', skip_header = True, fields = fields
)
(train_dataset, val_dataset, test_dataset) = dataset.split(split_ratio=[0.8, 0.1, 0.1])
"""
print("______________________________________________________")
# print("Number of train data: {}".format(len(train_dataset)))
# print("Number of test data: {}".format(len(test_dataset)))
# print("Number of validation data: {}".format(len(val_dataset)))
print("An Example: ", vars(train_dataset.examples[0]))
# {'label': '1', 'text': ['@kellbell68', 'yes', 'kelly', 'i', 'think', 'your', 'nice', 'but', 'come', 'on', 'your', 'from', 'ohio']}
print("______________________________________________________")

############# 修改词向量
# 加载glove词向量
# cache_dir是保存golve词典的缓存路径
# cache_dir = ''
# dim是embedding的维度
glove_vectors = Vectors(name='./Glove_Vectors/glove.6B/glove.6B.100d.txt')
############# 修改词向量

text_field.build_vocab(train_dataset, val_dataset,
                       min_freq = 5,
                       max_size = 10000,
                       vectors = glove_vectors,
                       unk_init = torch.Tensor.normal_
                       )
label_field.build_vocab(train_dataset, val_dataset)
# Most frequent tokens
# print("Most frequent tokens: ", text_field.vocab.freqs.most_common(10))

print("__________________________________________________")
# print("查看词向量,词表的size，就是有多少词：", len(text_field.vocab))
# print("查看词表中词向量的维度：", text_field.vocab.vectors.shape)
print("__________________________________________________")
# print(f"Unique tokens in TEXT vocabulary: {len(text_field.vocab)}")  # 25002
# print(f"Unique tokens in LABEL vocabulary: {len(label_field.vocab)}")  # 2 ? 4
# print(text_field.vocab.itos[:10])  # ['<unk>', '<pad>', 'the', ',', '.', 'and', 'a', 'of', 'to', 'is']
# print(label_field.vocab.stoi)  # defaultdict(None, {'neg': 0, 'pos': 1})
print("__________________________________________________")
################## 调整句子长度

train_iter, val_iter, test_iter = data.BucketIterator.splits((train_dataset, val_dataset, test_dataset),
                                             batch_sizes = (args.batch_size, args.batch_size, args.batch_size),
                                             sort_key = lambda x: len(x.sentence), ##############
                                             # sort_within_batch = True, ##############
                                             device = device
                                             )
################## 调整句子长度

embed_num = len(text_field.vocab) # vocabulary size
embed_dim = text_field.vocab.vectors.size()[-1] # vocabulary size
# class_num = len(label_field.vocab) - 1
print("-------------------------------------------")
print("词汇表大小/embed_num 是: ", embed_num)
print("词向量维度/embed_dim 是: 是否是100", embed_dim)
print("-------------------------------------------")

kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

args.cuda = torch.cuda.is_available()
# print("Parameters:")
#
# for attr, value in sorted(args.__dict__.items()):
#     print("{}={}".format(attr.upper(), value))
################################################################# Dataset #################################################################
################################################################# Dataset #################################################################

####### GitHub
# cnn = TextCnn(embed_num, args.embed_dim,
#               class_num, args.kernel_num,
#               kernel_sizes, args.dropout
#               )
####### GitHub

####### BiLSTM
# cnn = BiLSTM(vocab_size = embed_num,
#              embedding_dim = text_field.vocab.vectors.size()[-1], # 词向量维度 # dim must be equal to the dim of pre-trained GloVe vectors
#              hidden_dim = 256
#              ) # Get pad token index from vocab
####### BiLSTM

####### LSTM
# cnn = LSTM(vocab_size = embed_num,
#              embedding_dim = 100, # 词向量维度 # dim must be equal to the dim of pre-trained GloVe vectors
#              hidden_dim = 256, output_dim = 1, n_layers = 2,
#              bidirectional = True, dropout = 0.5,
#              pad_idx = text_field.vocab.stoi[text_field.pad_token])
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################################# Model #################################################################
################################################################# Model #################################################################
####### LSTM
# 定义分类类别
class_num = 6
# 构建模型
bilstm = BiLSTM_No_Length(vocab_size=embed_num, embedding_size=embed_dim, output_size=class_num)
# 加载预训练词向量
pretrained_embeddings = text_field.vocab.vectors
bilstm.embeddings.weight.data.copy_(pretrained_embeddings)  # 使用训练好的词向量初始化embedding层
# Get pad token index from vocab
####### LSTM


if args.snapshot is not None:
    print('Loading model from {}...'.format(args.snapshot))
    bilstm.load_state_dict(torch.load(args.snapshot))

# pytorch_total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
# print ("Model parameters: " + str(pytorch_total_params))


if args.cuda:
    cnn = bilstm.cuda()


if args.train:
    print("开始train过程")
    train(train_iter, val_iter, bilstm, args)


if args.test:
    print("start Test")
    bilstm.load_state_dict(torch.load("./snapshot_English/20231215/acc_best.pt")) # f1_best.pt
    print("load success!")
    test(test_iter, bilstm)


if args.predict:
    while(True):
        text = input(">>")
        label = predict(text, bilstm, text_field, label_field, False)
        print (str(label) + " | " + text)


