
import torch
from transformers import AutoTokenizer, AutoConfig
from modeling import (
    BertForSentimentClassification,
)
import numpy as np
import random
import os
import pandas as pd
import traceback


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


class Analyzer_prediction:
    def __init__(self, will_train, output_dir="./output_dir"):

        # If no model name/path is given, use mine/BERT depending on task.
        if will_train:
            self.model_name_or_path = "./model/bert-base-cased"  ##不区分大小写文本
            # args.model_name_or_path = "./model/bert-base-uncased"  ##只适用于小写文本
            # args.model_name_or_path = "./model/bert-base-chinese"  ##只适用于中文
            print("成功加载：", self.model_name_or_path)
        else:
            # args.model_name_or_path = "./models_6Classes_EN/20231215/best_acc_model"
            # self.model_name_or_path = r"F:\Python_Files\Social_Media_Data\sentiment_analysis\SA_6Classes\models_6Classes_EN\20231221-googlehuggingfacemerge\best_f1score_model"
            self.model_name_or_path = "./model/best_f1score_model"
            print("成功加载：", self.model_name_or_path)

        # Set up configuration.
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)

        # Create the model with the given configuration.
        if self.config.model_type == "bert":
            self.model = BertForSentimentClassification.from_pretrained(
                self.model_name_or_path
            )
            print("成功加载：", self.config.model_type)
        # elif self.config.model_type == "albert":
        #     self.model = AlbertForSentimentClassification.from_pretrained(
        #         model_name_or_path
        #     )
        # elif self.config.model_type == "distilbert":
        #     self.model = DistilBertForSentimentClassification.from_pretrained(
        #         model_name_or_path
        #     )
        else:
            raise ValueError("This transformer model is not supported yet.")

        # Set up device as GPU if available, otherwise CPU.

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Put model to device.
        self.model = self.model.to(self.device)

        # Set model to evaluation mode.
        self.model.eval()

        # Initialize tokenizer for the desired transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        # Set output directory.
        self.output_dir = output_dir



    def classify_sentiment(self, text):  # 对输出的情感概率进行分类
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
            sentiment_logit = self.model(
                input_ids = input_ids, attention_mask=attention_mask
            )
            sentiment_logit = sentiment_logit
            print("sentiment_logit: ", sentiment_logit)
            sentiment_logit_softmax = torch.nn.Softmax(dim=1)(sentiment_logit)
            print("sentiment_logit_squeeze: ", sentiment_logit_softmax)
            # 获取预测输出的标签
            pred_labels = torch.max(sentiment_logit_softmax, 1)[1]
            numpy_label = pred_labels.cpu().numpy()
            print("pred_label：", numpy_label)

            if numpy_label == 0:
                prediction = "sadness"
            elif numpy_label == 1:
                prediction = "joy"
            elif numpy_label == 2:
                prediction = "love"
            elif numpy_label == 3:
                prediction = "anger"
            elif numpy_label == 4:
                prediction = "fear"
            elif numpy_label == 5:
                prediction = "surprise"

            # if numpy_label == 0:
            #     prediction = "anger"
            # elif numpy_label == 1:
            #     prediction = "fear"
            # elif numpy_label == 2:
            #     prediction = "joy"
            # elif numpy_label == 3:
            #     prediction = "sadness"
            # elif numpy_label == 4:
            #     prediction = "surprise"
            # elif numpy_label == 5:
            #     prediction = "neural"
            return prediction

            # print("sentiment_probability：", positive_probability)
            #####################################################
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
            ######################################################


if __name__ == "__main__":

    analyzer_p = Analyzer_prediction(will_train=False)

    df = pd.read_excel("./test_excel.xlsx",
                              sheet_name="Sheet1", dtype=str)
    # df = pd.DataFrame(df)
    # df.insert(loc=14, column="prediction", value=np.nan)

    for i in range(len(df)):
        try:
            english_text = df.loc[i, "translate_to_english"]
            sentiment_classification = analyzer_p.classify_sentiment(english_text)
            df.loc[i, "prediction"] = sentiment_classification
            print(sentiment_classification)
            print("已完成"+ str(i+1) +"条")
        except Exception as e:
            # 异常发生时的处理代码
            traceback.print_exc()
            df.to_excel("./prediction_0123456-backup.xlsx", index=False)
            continue

    # test_samples1 = 'I love you very much!'
    # test_samples2 = 'This is a very bad man!'
    # Analyzer_prediction.classify_sentiment(test_samples1)
    print("全部运行结束！")
    # 保存所有数据
    df.to_excel("./prediction_0123456-success.xlsx", index=False)





