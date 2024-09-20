import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
import torch
import random
import numpy as np
import os
from dataset import SSTDataset
from arguments import args
from analyzer import Analyzer
from torch.utils.tensorboard import SummaryWriter

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

if __name__ == "__main__":

    # Initialize analyzer.
    analyzer = Analyzer(will_train=True, args=args)

    # Set citerion, which takes as input logits of positive class and computes binary cross-entropy.
    # criterion = nn.BCEWithLogitsLoss() # 二进制交叉熵损失函数——二分类
    criterion = nn.CrossEntropyLoss() # 损失函数修改为多分类，计算模型输出与目标标签之间的交叉熵损失，用于衡量模型的预测与真实标签之间的差异

    # 训练时冻结bert的前六层参数
    # for name, param in self.model.named_parameters():
    #     print(name, param.shape)
    # param.requires_grad = False
    # freeze_layers = ("conv1", "conv2")
    # for name, param in model.named_parameters():
    #     print(name, param.shape)
    #     if name.split(".")[0] in freeze_layers:
    #         param.requires_grad = False
    # 加载并冻结bert模型参数
    # for name, param in analyzer.model.named_parameters():
    #     if name.startswith('pooler'):
    #         continue
    #     else:
    #         param.requires_grad = False

    # Set optimizer to Adam.
    # optimizer = optim.Adam(params=analyzer.model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(params=filter(lambda p : p.requires_grad, analyzer.model.parameters()), lr=args.lr) # 只传入需要更新梯度的参数
    optimizer = optim.AdamW(params=analyzer.model.parameters(), lr=args.lr)

    # Initialize training set and loader.
    train_set = SSTDataset(
        filename = "data/English_6classes/merge_google_huggingface_6classes/google_emotion_class_average6_train.csv", ##
        # filename = "data/English_6classes/emotion_dataset_6classes_train.csv",
        maxlen = args.maxlen_train,
        tokenizer = analyzer.tokenizer,
    )
    val_set = SSTDataset(
        filename = "data/English_6classes/merge_google_huggingface_6classes/google_emotion_class_average6_val.csv", ##
        # filename = "data/English_6classes/emotion_dataset_6classes_val.csv",
        maxlen=args.maxlen_val,
        tokenizer=analyzer.tokenizer
    )

    # Initialize validation set and loader.
    train_loader = DataLoader(
        dataset = train_set,
        batch_size = args.batch_size,
        num_workers = args.num_threads,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset = val_set,
        batch_size = args.batch_size,
        num_workers = args.num_threads,
        shuffle=True
    )

    # emotion_list = ['anger', 'fear', 'joy', 'neural', 'sadness', 'surprise']
    # label_dict = {  'anger': 0,
    #                 'fear': 1,
    #                 'joy': 2,
    #                 'neural': 3,
    #                 'sadness': 4,
    #                 'surprise': 5
    #               }

    # Initialize best accuracy.
    best_accuracy = 0
    best_f1score = 0
    # Go through epochs.
    # 确定epoch loss保存的路径，会保存一个文件夹，而非文件
    writer = SummaryWriter("logs_train_BERT_6Classes_EN")

    for epoch in trange(args.num_eps, desc="Epoch"):
        print("-------第{}轮训练开始-------".format(epoch + 1))

        # Train analyzer for one epoch.
        train_accuracy, train_avg_loss = analyzer.train(
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion)

        # Evaluate analyzer; get validation loss and accuracy.
        # val_accuracy, val_avg_loss, val_f1score = analyzer.evaluate(
        val_accuracy, val_avg_loss, val_f1score, val_precision, val_recall, val_bcm = analyzer.evaluate(
            val_loader=val_loader,
            criterion=criterion)


        # Display validation accuracy and loss.
        print(
            f"Epoch {epoch} complete! "
            f"Train Accuracy (6 Classes English) : {train_accuracy}, "
            f"Train Loss (6 Classes English) : {train_avg_loss}, "
            f"Val Accuracy (6 Classes English) : {val_accuracy}, "
            f"Val Loss (6 Classes English) : {val_avg_loss}, "
            f"Val F1_Score (6 Classes English) : {val_f1score}, "
            f"Val Precision (6 Classes English) : {val_precision}, "
            f"Val Recall (6 Classes English) : {val_recall} ,"
            f"Val ConfusionMatrix (6 Classes English) : {val_bcm} "
        )

        # tensorboard 写入
        writer.add_scalars('loss',
                           {'train_avg_loss_6Classes_EN': train_avg_loss,
                                        'val_avg_loss_6Classes_EN': val_avg_loss}, epoch)  # 第三步，绘图
        writer.add_scalar("val_accuracy_6Classes_EN", val_accuracy, epoch)  # 第三步，绘图
        writer.add_scalar("val_F1-Score_6Classes_EN", val_f1score, epoch)  # 第三步，绘图
        writer.add_scalar("val_Precision_6Classes_EN", val_precision, epoch)  # 第三步，绘图
        writer.add_scalar("val_Recall_6Classes_EN", val_recall, epoch)  # 第三步，绘图


        # Save analyzer if validation accuracy imporoved.
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(
                f"Best validation accuracy improved from "
                f"{best_accuracy} to {val_accuracy}, saving analyzer..."
            )
            analyzer.save_acc()
        # Save analyzer if validation f1_score imporoved.
        if val_f1score > best_f1score:
            best_f1score = val_f1score
            print(
                f"Best validation f1score improved from "
                f"{best_f1score} to {val_f1score}, saving analyzer..."
            )
            analyzer.save_f1score()


    writer.close()  # 第4步，写入关闭
