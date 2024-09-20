import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SSTDataset
from arguments import args
from analyzer import Analyzer


if __name__ == "__main__":

    # Initialize analyzer.
    analyzer = Analyzer(will_train=False, args=args)

    # Set citerion, which takes as input logits of positive class and computes binary cross-entropy.
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss() # 多分类

    # Initialize validation set and loader.
    test_set = SSTDataset(
        # filename="data/dev.tsv", maxlen=args.maxlen_val, tokenizer=analyzer.tokenizer
        # filename="data/English_6classes/emotion_dataset_6classes_test.csv", maxlen=args.maxlen_val, tokenizer=analyzer.tokenizer
        filename="data/English_6classes/merge_google_huggingface_6classes/google_emotion_class_average6_test.csv", maxlen=args.maxlen_val, tokenizer=analyzer.tokenizer
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, num_workers=args.num_threads
    )

    # Evaluate analyzer and get accuracy + loss.
    test_total_acc, test_avg_loss, test_total_f1score, test_total_precision, test_total_recall, test_total_bcm = analyzer.evaluate(
        val_loader=test_loader, criterion=criterion
    )

    # Display accuracy and loss.
    print(
        f"test Loss (6 Classes English) : {test_avg_loss}, "
        f"test Accuracy (6 Classes English) : {test_total_acc}, "
        f"test F1_Score (6 Classes English) : {test_total_f1score}, "
        f"test Precision (6 Classes English) : {test_total_precision}, "
        f"test Recall (6 Classes English) : {test_total_recall},"
        f"test ConfusionMatrix (6 Classes English) : {test_total_bcm}"
    )