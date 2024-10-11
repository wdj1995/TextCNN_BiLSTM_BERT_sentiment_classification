import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 随机森林相关的包
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

# 设置显示所有列
pd.set_option('display.max_columns', None)
# 设置显示所有行
pd.set_option('display.max_rows', None)

# 设置随机种子以确保结果可复现
np.random.seed(123)

def k_fold_cross_validation(n_estimators):
    df = pd.read_excel('./193countries_emotion_index_delete_no_tweets_RandomForest.xlsx', sheet_name='Sheet1')
    rf_df = df[["country_code", "region_label", "total_population_percentage",
                "religion_percentage", "GDP_percentage", "education", "Internet"]]
    # print(rf_df)
    # 随机森林数据集划分
    # 假设第一列是标签列
    X = rf_df.iloc[:, 2:]  # 特征
    y = rf_df.iloc[:, 1]  # 标签
    # 划分训练集和测试集 # 多折交叉验证不需要划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    # 构建随机森林模型
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=123, n_jobs=-1)

    # ########################################## 多折交叉验证 ##########################################
    # # 交叉验证
    # # 定义评价指标
    # scoring = {
    #     'accuracy': make_scorer(accuracy_score),
    #     'precision': make_scorer(precision_score, average='weighted'),
    #     'recall': make_scorer(recall_score, average='weighted'),
    #     'f1': make_scorer(f1_score, average='weighted')
    # }
    # kf = KFold(n_splits=5, shuffle=True, random_state=123) # 10折交叉验证
    # cv_results = cross_validate(estimator=clf,
    #                             X=X, y=y,
    #                             cv=kf,
    #                             scoring=scoring,
    #                             return_train_score=False, # # 返回训练集上的得分
    #                             n_jobs=-1 # # 使用所有可用的核心进行并行处理
    #                             )
    # # 输出结果
    # print("\nAverage testing scores:")
    # print("Accuracy: %0.4f (+/- %0.4f)" % (cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std() * 2))
    # print("Precision (weighted): %0.4f (+/- %0.4f)" % (cv_results['test_precision'].mean(),
    #                                                    cv_results['test_precision'].std() * 2))
    # print("Recall (weighted): %0.4f (+/- %0.4f)" % (cv_results['test_recall'].mean(),
    #                                              cv_results['test_recall'].std() * 2))
    # print("F1 Score (weighted): %0.4f (+/- %0.4f)" % (cv_results['test_f1'].mean(),
    #                                                   cv_results['test_f1'].std() * 2))
    # print("__________________________________________________________________________")
    # print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
    # print(f"Precision: {np.mean(cv_results['test_precision']):.4f}")
    # print(f"Recall: {np.mean(cv_results['test_recall']):.4f}")
    # print(f"F1-Score: {np.mean(cv_results['test_f1']):.4f}")
    # print("__________________________________________________________________________")
    # ########################################## 多折交叉验证 ##########################################

    ########################################## 计算特征重要性 ##########################################
    # # 训练模型
    # 如果你在评估一个模型时计算特征重要性（例如使用所有数据训练一个模型来查看哪些特征对该模型最重要），此时你可以使用全部数据进行计算，因为此时你的目标是了解模型对特征的依赖性。
    clf.fit(X_train, y_train)
    # 计算准确率
    baseline_accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f'Baseline Accuracy: {baseline_accuracy:.4f}')

    # MDA需要划分训练集和测试集
    # MDA是一种更稳定的特征重要性度量方法，因为它不受特征的内在性质影响，并且更能反映特征在预测中的实际重要性。
    mda_importance = permutation_importance(clf, X_test, y_test, n_repeats=100, random_state=123, n_jobs=-1)
    # mda_importance_perm = mda_importance.importances_mean
    # 打印特征重要性
    # indices_perm = np.argsort(mda_importance_perm)[::-1]  # 按重要性降序排列的索引
    # for f in range(X.shape[1]):
    #     print(f"{f + 1}. feature_mda {X.columns[indices_perm[f]]} ({mda_importance_perm[indices_perm[f]]})")
    # 输出特征重要性
    for i in mda_importance.importances_mean.argsort()[::-1]:
        print(f"{X.columns[i]}: {mda_importance.importances_mean[i]:.4f} +/- {mda_importance.importances_std[i]:.4f}")
    ########################################## 计算特征重要性 ##########################################
    return baseline_accuracy, mda_importance.importances_mean[0]


if __name__ == '__main__':

    ########################################################################
    # 树的个数为15
    # Accuracy: 0.7273
    # GDP_percentage: 0.2634 + / - 0.0680
    # Internet: 0.0668 + / - 0.0452
    # total_population_percentage: 0.0255 + / - 0.0262
    # religion_percentage: -0.0061 + / - 0.0380
    # education: -0.0114 + / - 0.0330
    ########################################################################
    # 数的个数为13
    # Accuracy: 0.7500
    # GDP_percentage: 0.2568 + / - 0.0610
    # Internet: 0.1157 + / - 0.0459
    # total_population_percentage: 0.0359 + / - 0.0288
    # religion_percentage: 0.0045 + / - 0.0404
    # education: 0.0041 + / - 0.0324
    ########################################################################



    best_acc = 0
    acc_num = 0

    best_mda = 0
    mda_num = 0

    for i in range(5, 501):
        acc, mda = k_fold_cross_validation(i)
        if acc > best_acc:
            best_acc = acc
            acc_num = i
        if mda > best_mda:
            best_mda = mda
            mda_num = i
    print(f"Best Accuracy: {best_acc:.4f} with n_estimators={acc_num}")
    print(f"Best MDA: {best_mda:.4f} with n_estimators={mda_num}")

    # rf_classify_and_calculate_importance()