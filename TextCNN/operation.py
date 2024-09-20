import os
import sys
import torch
import numpy as np
import random
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.classification import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassConfusionMatrix
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_iter, dev_iter, model, args):

    model.train()
    # 定义优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # 损失函数
    # criterion = nn.BCEWithLogitsLoss()  # 二进制交叉熵损失函数——二分类
    criterion = nn.CrossEntropyLoss()  # 多分类使用交叉熵损失函数

    steps = 0
    # best_acc = 0
    # last_step = 0
    best_accuracy = 0
    best_f1score = 0

    writer = SummaryWriter("logs_train_6Classes_English")

    for epoch in range(1, args.epochs+1):
        train_total_loss = 0

        train_acc = MulticlassAccuracy(num_classes=6).to(device)

        for i, batch in enumerate(train_iter):
            feature, target = batch.sentence, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()

            logit = model(feature)
            # print("logit : ", logit)
            # print("target : ", target)
            # loss = F.cross_entropy(logit, target) # 多分类损失函数
            loss = criterion(logit.squeeze(-1), target.long()) # 二分类损失函数

            # training step accuracy
            batch_acc = train_acc(logit.squeeze(-1), target.float())
            print(f"Train_batch_acc of Epoch:{epoch}_Step:{steps} is {batch_acc}")

            # Backpropagate the loss. # 反向传播
            loss.backward()
            # Optimize the model. # 更新参数
            optimizer.step()

            train_total_loss += loss.item()

            steps += 1
        print("{} epoch of steps is :".format(epoch), steps)
        train_total_acc = train_acc.compute()
        train_acc.reset()
        # 计算平均损失
        train_avg_loss = train_total_loss / len(train_iter.dataset)
        print(f"Train_avg_loss of epoch{epoch} is {train_avg_loss}")
        print(f"Train_total_acc of epoch{epoch} is {train_total_acc}")
        # return train_total_acc, train_avg_loss

        val_total_acc, val_avg_loss, val_total_f1score, val_total_precision, val_total_recall \
            = test(dev_iter, model, args)

        # tensorboard 写入
        writer.add_scalars('loss',
                           {'train_avg_loss_6Classes': train_avg_loss,
                            'val_avg_loss_6Classes': val_avg_loss}, epoch)
        writer.add_scalar("val_accuracy_6Classes", val_total_acc, epoch)
        writer.add_scalar("val_F1-Score_6Classes", val_total_f1score, epoch)
        writer.add_scalar("val_Precision_6Classes", val_total_precision, epoch)
        writer.add_scalar("val_Recall_6Classes", val_total_recall, epoch)

        # Save analyzer if validation accuracy imporoved.
        if val_total_acc > best_accuracy:
            best_accuracy = val_total_acc
            print(
                f"Best validation accuracy improved from "
                f"{best_accuracy} to {val_total_acc}, saving analyzer..."
            )
            save_acc(model, args.save_dir)

        # Save analyzer if validation f1_score imporoved.
        if val_total_f1score > best_f1score:
            best_f1score = val_total_f1score
            print(
                f"Best validation f1score improved from "
                f"{best_f1score} to {val_total_f1score}, saving analyzer..."
            )
            save_f1(model, args.save_dir)

    writer.close()  # tensorboard写入关闭


def test(test_data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0

    # 损失函数
    # criterion = nn.BCEWithLogitsLoss()  # 二进制交叉熵损失函数——二分类
    criterion = nn.CrossEntropyLoss()  # 多分类使用交叉熵损失函数

    val_total_loss = 0

    # 计算accuracy指标
    val_acc = MulticlassAccuracy(num_classes=6, average='weighted').to(device)  # 计算accuracy
    # 计算F1-score指标
    val_f1score = MulticlassF1Score(num_classes=6, average='weighted').to(device)
    # 计算Precision指标
    val_precision = MulticlassPrecision(num_classes=6, average='weighted').to(device)
    # 计算Recall指标
    val_recall = MulticlassRecall(num_classes=6, average='weighted').to(device)
    # 计算混淆矩阵
    val_bcm = MulticlassConfusionMatrix(num_classes=6).to(device)

    # Don't track gradient.
    with torch.no_grad():
        for batch in test_data_iter:
            feature, target = batch.sentence, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit = model(feature)

            loss = criterion(logit.squeeze(-1), target.long())  # 二分类交叉熵损失函数

            val_total_loss += loss.item()
            # corrects += (torch.max(logit, 1)
            #              [1].view(target.size()).data == target.data).sum()

            # 计算 accuracy
            val_acc.update(logit.squeeze(-1), target.float())  ##.argmax(1)
            # print("Val accuracy on batch is ", val_batch_acc)
            # 计算F1-Score
            val_f1score.update(logit.squeeze(-1), target.float())  ##.argmax(1)
            # print("Val F1-Score on batch is ", val_batch_f1)
            # 计算Precision
            val_precision.update(logit.squeeze(-1), target.float())  ##.argmax(1)
            # 计算recall
            val_recall.update(logit.squeeze(-1), target.float())  ##.argmax(1)
            # 计算混淆矩阵
            val_bcm.update(logit.squeeze(-1), target.float())

    # 计算平均损失
    val_avg_loss = val_total_loss / len(test_data_iter.dataset)
    # Calculate accuracy.
    # accuracy = batch_accuracy_summation / num_batches ### origin
    # 计算accuracy
    val_total_acc = val_acc.compute()  # torchmetrics计算accuracy
    print("val_total_Accuracy is ", val_total_acc)
    val_acc.reset()
    # 计算F1-score
    val_total_f1score = val_f1score.compute()  # torchmetrics计算accuracy
    print("val_total_F1Score is ", val_total_f1score)
    val_f1score.reset()
    # 计算F1-score
    val_total_precision = val_precision.compute()  # torchmetrics计算accuracy
    print("val_total_Precision is ", val_total_precision)
    val_precision.reset()
    # 计算Recall
    val_total_recall = val_recall.compute()  # torchmetrics计算accuracy
    print("val_total_Recall is ", val_total_recall)
    val_recall.reset()

    # 计算混淆矩阵
    val_total_bcm = val_bcm.compute()  # torchmetrics计算accuracy
    print("val_total_ConfusionMatrix is ", val_total_bcm)
    val_bcm.reset()

    # size = len(test_data_iter.dataset)
    # avg_loss /= size
    # accuracy = 100.0 * float(corrects) / size
    # print('Evaluation - loss: {:.6f}  acc: {:.3f}% ({}/{}) \n'.format(avg_loss,
    #                                                                     accuracy,
    #                                                                     corrects,
    #                                                                     size))
    # return accuracy
    return val_total_acc, val_avg_loss, val_total_f1score, val_total_precision, val_total_recall


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()

    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()

    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data + 1]


def save_acc(model, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'acc_best.pt')
    torch.save(model.state_dict(), save_path)

def save_f1(model, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'f1_best.pt')
    torch.save(model.state_dict(), save_path)


