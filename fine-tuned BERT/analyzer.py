import torch
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
from modeling import (
    BertForSentimentClassification,
    AlbertForSentimentClassification,
    DistilBertForSentimentClassification,
)
from utils import get_accuracy_from_logits
import numpy as np
import random
import os
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
# from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassConfusionMatrix

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

class Analyzer:
    def __init__(self, will_train, args):

        # If no model name/path is given, use mine/BERT depending on task.
        if args.model_name_or_path is None:
            if will_train:
                args.model_name_or_path = "./model/bert-base-cased"  ##不区分大小写文本
                # args.model_name_or_path = "./model/bert-base-uncased"  ##只适用于小写文本
                # args.model_name_or_path = "./model/bert-base-chinese"  ##只适用于中文
                print("成功加载：", args.model_name_or_path)
            else:
                # args.model_name_or_path = "./models_6Classes_EN/20231215/best_acc_model"
                args.model_name_or_path = "./models_BERT_6Classes_EN/best_f1score_model"
                print("成功加载：", args.model_name_or_path)

        # Set up configuration.
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)

        # Create the model with the given configuration.
        if self.config.model_type == "bert":
            self.model = BertForSentimentClassification.from_pretrained(
                args.model_name_or_path
            )
            print("成功加载：", self.config.model_type)
        elif self.config.model_type == "albert":
            self.model = AlbertForSentimentClassification.from_pretrained(
                args.model_name_or_path
            )
        elif self.config.model_type == "distilbert":
            self.model = DistilBertForSentimentClassification.from_pretrained(
                args.model_name_or_path
            )
        else:
            raise ValueError("This transformer model is not supported yet.")

        # Set up device as GPU if available, otherwise CPU.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Put model to device.
        self.model = self.model.to(self.device)

        # Set model to evaluation mode.
        self.model.eval()

        # Initialize tokenizer for the desired transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        # Set output directory.
        self.output_dir = args.output_dir

    # def f1_score_func(preds, labels):
    #     preds_flat = np.argmax(preds, axis=1).flatten()
    #     labels_flat = labels.flatten()
    #     return f1_score(labels_flat, preds_flat, average='weighted')


    # Trains analyzer for one epoch.
    def train(self, train_loader, optimizer, criterion):
        # Set model to training mode.
        self.model.train()
        # # 清空一下cuda缓存
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        # Go through training set in batches.
        train_total_loss = 0

        train_acc = MulticlassAccuracy(num_classes = 6).to(self.device)

        for input_ids, attention_mask, labels in tqdm(
            iterable=train_loader, desc="Training"):

            # Reset gradient
            optimizer.zero_grad()
            # Put input IDs, attention mask, and labels to device
            input_ids, attention_mask, labels = (
                input_ids.to(self.device),
                attention_mask.to(self.device),
                labels.to(self.device),
            )
            #####################################################
            # Get logits.
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Get loss. # 计算损失
            loss = criterion(input=logits.squeeze(-1), target=labels.long()) # float->long #
            # loss = criterion(input=logits, target=labels.long()) # float->long #
            # print("train output loss: ", loss)

            batch_train_acc = train_acc(logits.squeeze(-1), labels.long())
            print("Train accuracy on batch is ", batch_train_acc)

            # Backpropagate the loss. # 反向传播
            loss.backward()
            # Optimize the model. # 更新参数
            optimizer.step()

            train_total_loss += loss.item()

        train_total_acc = train_acc.compute()
        train_acc.reset()
        # 计算平均损失
        # print("len(train_loader): ",len(train_loader))
        train_avg_loss = train_total_loss / len(train_loader)
        return train_total_acc, train_avg_loss


    # Evaluates analyzer.
    def evaluate(self, val_loader, criterion):
        # Set model to evaluation mode.
        self.model.eval()
        # Initialize batch accuracy summation, loss, and number of batches.
        batch_accuracy_summation, loss, num_batches = 0, 0, 0
        val_total_loss = 0

        # 计算accuracy指标
        val_acc = MulticlassAccuracy(num_classes=6, average='weighted').to(self.device)  # 计算accuracy
        # 计算F1-score指标
        val_f1score = MulticlassF1Score(num_classes=6, average='weighted').to(self.device)
        # 计算Precision指标
        val_precision = MulticlassPrecision(num_classes=6, average='weighted').to(self.device)
        # 计算Recall指标
        val_recall = MulticlassRecall(num_classes=6, average='weighted').to(self.device)
        # 计算混淆矩阵
        val_bcm = MulticlassConfusionMatrix(num_classes=6).to(self.device)


        # Don't track gradient.
        with torch.no_grad():
            # Go through validation set in batches.
            for input_ids, attention_mask, labels in tqdm(
                    val_loader, desc="Evaluating"):
                # Put input IDs, attention mask, and labels to device.
                input_ids, attention_mask, labels = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    labels.to(self.device),
                )
                # Get logits.
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                #######################################################################
                # Get batch accuracy and add it. ### 需要修改，不需要sigmoid
                # batch_accuracy_summation += get_accuracy_from_logits(logits, labels)

                # Get batch loss and add it.
                loss = criterion(logits.squeeze(-1), labels.long())  # float->long logits.squeeze(-1)#
                # loss += criterion(logits.squeeze, labels.long()).item() # float->long
                val_total_loss += loss.item()
                # Increment num_batches.
                # num_batches += 1 ####

                # 计算 accuracy
                val_acc.update(logits.squeeze(-1), labels.float())
                # print("Val accuracy on batch is ", val_batch_acc)
                # 计算F1-Score
                val_f1score.update(logits.squeeze(-1), labels.float())
                # print("Val F1-Score on batch is ", val_batch_f1)
                # 计算Precision
                val_precision.update(logits.squeeze(-1), labels.float())  ##.argmax(1)
                # 计算recall
                val_recall.update(logits.squeeze(-1), labels.float())  ##.argmax(1)
                # 计算混淆矩阵
                val_bcm.update(logits.squeeze(-1), labels.float())


        # 计算平均损失
        val_avg_loss = val_total_loss / len(val_loader)
        # Calculate accuracy.
        # accuracy = batch_accuracy_summation / num_batches ### origin
        # 计算accuracy
        val_total_acc = val_acc.compute()  # torchmetrics计算accuracy
        print("val_total_acc is ", val_total_acc)
        val_acc.reset()
        # 计算F1-score
        val_total_f1score = val_f1score.compute()  # torchmetrics计算accuracy
        print("val_total_f1score is ", val_total_f1score)
        val_f1score.reset()
        # 计算precision
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


        # Return accuracy and loss.
        # return accuracy.item(), loss ##
        return val_total_acc, val_avg_loss, val_total_f1score, val_total_precision, val_total_recall, val_total_bcm


    # Saves analyzer.
    def save_acc(self):
        # Save model.
        self.model.save_pretrained(save_directory=f"models_BERT_6Classes_EN/best_acc_model/")
        # Save configuration.
        self.config.save_pretrained(save_directory=f"models_BERT_6Classes_EN/best_acc_model/")
        # Save tokenizer.
        self.tokenizer.save_pretrained(save_directory=f"models_BERT_6Classes_EN/best_acc_model/")
    # Saves analyzer.
    def save_f1score(self):
        # Save model.
        self.model.save_pretrained(save_directory=f"models_BERT_6Classes_EN/best_f1score_model/")
        # Save configuration.
        self.config.save_pretrained(save_directory=f"models_BERT_6Classes_EN/best_f1score_model/")
        # Save tokenizer.
        self.tokenizer.save_pretrained(save_directory=f"models_BERT_6Classes_EN/best_f1score_model/")


    # 后面要进行下游的任务，我们把这个预训练模型的前 10 层都冻结住
    def freeze_unfreeze_layers(model, layer_indexs, unfreeze=False):
        if type(layer_indexs) == int:
            freeze_layer = model.layer[layer_indexs]
            for para in freeze_layer.parameters():
                para.requires_grad_(unfreeze)
            print(f"successfully freeze layers index: {layer_indexs}")

        else:
            start = layer_indexs[0]
            end = layer_indexs[1]
            freeze_layers = model.layer[start: end + 1]
            for la in freeze_layers:
                for para in la.parameters():
                    # 这里的函数使用下划线，直接覆盖原值
                    para.requires_grad_(unfreeze)
            print(
                f"successfully freeze layers indexs from: {layer_indexs[0]} to: {layer_indexs[1]}, including {layer_indexs[1]}")


    # Classifies sentiment as positve or negative.
    def classify_sentiment(self, text): # 对输出的情感概率进行分类
        # Don't track gradient.
        with torch.no_grad():
            # Tokens are made up of CLS token, text converted to tokens, and SEP token.
            tokens = ["[CLS]"] + self.tokenizer.tokenize(text) + ["[SEP]"]
            # Convert tokens to input IDs; convert them to tensor, unsqueeze, put it to device.
            input_ids = (
                torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
                .unsqueeze(0)
                .to(self.device)
            )
            # Create attention mask from input IDs.
            attention_mask = (input_ids != 0).long()
            # Get logit (log-odds) of sentiment being positive from the model.
            positive_logit = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            print("positive_logit: ", positive_logit)

            ######################################################
            # Convert the logit to a probability.
            # positive_probability = torch.sigmoid(positive_logit.unsqueeze(-1)).item()

            # Convert the probability to a percentage.
            # positive_percentage = positive_probability * 100
            # Conver probability to boolean.
            # is_positive = positive_probability > 0.5
            # Return sentiment and percentage.
            # if is_positive:
            #     return "Positive", int(positive_percentage)
            # else:
            #     return "Negative", int(100 - positive_percentage)


    def accuracy_per_class(preds, labels):
        emotion_list = ['anger', 'fear', 'joy', 'neural', 'sadness', 'surprise']
        label_dict = {'anger': 0,
                      'fear': 1,
                      'joy': 2,
                      'neural': 3,
                      'sadness': 4,
                      'surprise': 5
                      }
        label_dict_inverse = {v: k for k, v in label_dict.values()}
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')

