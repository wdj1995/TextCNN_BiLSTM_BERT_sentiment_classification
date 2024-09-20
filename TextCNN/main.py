import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as data
import torchtext.datasets as datasets
import jieba
import argparse
from torchtext import data
from model import TextCNN, TextCNNCSDN
# from model import TextCNNCSDN
from operation import *
from torchtext.vocab import Vectors
import torchtext


######################### set Random Seed
torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)
os.environ['PYTHONHASHSEED'] = str(3407)  # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(3407)     # 为CPU设置随机种子
torch.cuda.manual_seed(3407)      # 为当前GPU设置随机种子（只用一块GPU）
# torch.cuda.manual_seed_all(3407)   # 为所有GPU设置随机种子（多块GPU）
torch.backends.cudnn.deterministic = True
######################### set Random Seed


def parse_arguments():
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=150, help='number of epochs for train [default: 10]')
    parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
    parser.add_argument('-log-interval',  type=int, default=100,   help='how many steps to wait before logging training status [default: 100]')
    parser.add_argument('-test-interval', type=int, default=200, help='how many steps to wait before testing [default: 200]')
    parser.add_argument('-save-interval', type=int, default=1000, help='how many steps to wait before saving [default: 1000]')
    parser.add_argument('-save-dir', type=str, default='snapshot_English', help='directory to save the snapshot')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='dropout probability [default: 0.5]')
    # parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
    # parser.add_argument('-kernel-num', type=int, default=10, help='number of kernels')
    parser.add_argument('-kernel-num', type=int, default=16, help='number of kernels')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-train', action='store_true', default=False, help='train a new model')
    parser.add_argument('-test', action='store_true', default=False, help='test on testset, combined with -snapshot to load model')
    parser.add_argument('-predict', action='store_true', default=False, help='predict label of console input')
    args = parser.parse_args()

    return args


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
text_field = data.Field(lower=True, tokenize = tokenize_EN, stop_words=stop_words)
label_field = data.Field(sequential=False) # sequential是否把数据表示成序列,如果是False,不能使用分词,默认值: True.

# fields = [('text', text_field), ('label', label_field)]
fields = [('sentence', text_field), ('label', label_field)]

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
# print(train_dataset)
"""
############# 修改词向量
# pretrained_name = 'zhwiki_2017_03.sg_50d.word2vec' # 预训练词向量文件名
# pretrained_path = './ChineseWordsVectors/' #预训练词向量存放路径
# vectors_CN = torchtext.vocab.Vectors(name = pretrained_name, cache = pretrained_path)
# print("vectors_CN: ",vectors_CN.dim)
# word2vec加载
# word2vec_model = Word2Vec.load(pretrained_path + pretrained_name)
# word_vecs = torch.FloatTensor(word2vec_model.wv.syn0)
# 通过Vocab在预先设置好的词汇表中（由Vectors导入）对文本数据进行向量化，然后按照Vocab的参数对数据集进行一系列操作。
# 建立词表
# text_field.build_vocab(train_dataset, val_dataset, min_freq = 5, max_size = 50000)
glove_vectors = Vectors(name='./Glove_Vectors/glove.6B/glove.6B.100d.txt')
############# 修改词向量

text_field.build_vocab(train_dataset, val_dataset,
                       min_freq = 5, max_size = 50000,
                       vectors = glove_vectors,
                       unk_init = torch.Tensor.normal_ # 给tensor初始化，一般是给网络中参数weight初始化，初始化参数值符合正态分布。
)
label_field.build_vocab(train_dataset, val_dataset)

# print("查看词向量,词表的size，就是有多少词：", len(text_field.vocab))
# print("查看词表中词向量的维度：", text_field.vocab.vectors.shape)
"""
train_iter, test_iter = data.Iterator.splits((train_dataset, val_dataset),
                                             batch_sizes = (args.batch_size, args.batch_size),
                                             sort_key = lambda x: len(x.sentence),
                                             sort_within_batch=False,
                                             shuffle=True,
                                             repeat=False
                                             )
"""
train_iter, val_iter, test_iter = data.BucketIterator.splits((train_dataset, val_dataset, test_dataset),
                                             batch_sizes = (args.batch_size, args.batch_size, args.batch_size),
                                             sort_key = lambda x: len(x.sentence), ##############
                                             # sort_within_batch = True, ##############
                                             device = device
                                             )

embed_num = len(text_field.vocab)
embed_dim = text_field.vocab.vectors.size()[-1] # vocabulary size
# class_num = len(label_field.vocab) - 1
print("-------------------------------------------")
print("词汇表大小/embed_num 是: ", embed_num)
print("词向量维度/embed_dim 是: 是否是100", embed_dim)
print("-------------------------------------------")

kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

args.cuda = torch.cuda.is_available()
# print("Parameters:")
# for attr, value in sorted(args.__dict__.items()):
#     print("{}={}".format(attr.upper(), value))

####### GitHub
############# 定义分类类别
class_num = 6
############# 定义分类类别
cnn = TextCNN(embed_num=embed_num, embed_dim=embed_dim,
              class_num=class_num, kernel_num=args.kernel_num,
              kernel_sizes=kernel_sizes
              )
####### GitHub

####### CSDN
# cnn = TextCNNCSDN(vocabulary_size = embed_num, #词表的大小
#                   # embedding_dimension = args.embed_dim, # 词向量的维度
#                   embedding_dimension = embed_dim, # 词向量维度, # 词向量的维度
#                   class_num = class_num,
#                   filter_num = args.kernel_num,
#                   filter_sizes = kernel_sizes,
#                   # vectors = glove_vectors, # 词向量
#                   dropout = args.dropout) # filter_sizes,filter_num,vocabulary_size,embedding_dimension,vectors,dropout
# 加载预训练词向量
pretrained_embeddings = text_field.vocab.vectors
cnn.embeddings.weight.data.copy_(pretrained_embeddings)  # 使用训练好的词向量初始化embedding层
# cnn.embed.weight.data.copy_(torch.Tensor(vectors))  # 使用训练好的词向量初始化embedding层
# cnn.embed.weight.requires_grad = False
####### CSDN

if args.snapshot is not None:
    print('Loading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))


# pytorch_total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
# print ("Model parameters: " + str(pytorch_total_params))


if args.cuda:
    cnn = cnn.cuda()


if args.train:
    train(train_iter, test_iter, cnn, args)


if args.test:
    print("start Test")
    cnn.load_state_dict(torch.load("./snapshot_English/20231214-1/f1_best.pt"))  # f1_best.pt
    print("load success!")
    test(test_iter, cnn, args)


if args.predict:
    while(True):
        text = input(">>")
        label = predict(text, cnn, text_field, label_field, False)
        print (str(label) + " | " + text)



