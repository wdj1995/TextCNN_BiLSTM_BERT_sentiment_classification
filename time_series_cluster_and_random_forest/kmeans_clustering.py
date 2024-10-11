import pandas as pd
import numpy as np
from datetime import datetime
from tslearn.clustering import silhouette_score
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# 设置显示选项
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整宽度
pd.set_option('display.max_colwidth', None)  # 显示完整的列宽
# 设置 NumPy 显示选项
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
def no_data_add_0(input_time, input_count, start_time, end_time): # input_time, input_count, start_time, end_time
    # 根据时间创建一天列表
    df_time = pd.date_range(start_time, end_time, freq="1d")
    df_time = df_time.strftime('%Y-%m-%d') # 转换成字符串类型
    new_time_arr = df_time.to_numpy()  #根据定义时间生成每一天
    for i in new_time_arr:
        if i in input_time:
            print("输入数组中存在该时间")
        if i not in input_time:
            print("输入数组中不存在该时间")
            input_time.append(i)
            input_count.append(0)
    output_time = input_time
    output_count = input_count
    return output_time, output_count

def adjust_time_correct_order(input_time, input_count):
    # 使用字典创建 DataFrame
    df = pd.DataFrame({'time': input_time, 'number': input_count})
    df['time'] = pd.to_datetime(df['time']) # 转换数据类型
    df_sorted = df.sort_values(by='time') # 对时间数据进行排序
    df_time = df_sorted["time"].dt.strftime('%Y-%m-%d')  # 转换成字符串类型
    output_time = df_time.to_numpy() # 转换成字符串类型
    output_count = df_sorted["number"].to_numpy()
    # 数字输出百分比
    # arr_sum = sum(output_count)
    # output_count = [((i/arr_sum)*100) for i in output_count]
    # output_count = [int(j*1000) / 1000 for j in output_count]
    return output_time, output_count

def get_time_count(df_0, start_time, end_time):
    df_0_time = df_0['Timestamp'].str.split('T', expand=True)[0] # 对时间列进行数据过滤，只保留日期格式
    df_0_time = pd.to_datetime(df_0_time) # 转换日期的格式
    df_0_time = df_0_time.dt.strftime('%Y-%m-%d') # 转换为年月日的格式
    df_0_time = df_0_time.sort_values(ascending=True)  # 升序排列
    df_0_time = df_0_time.drop_duplicates()  # 过滤，去除重复值
    df_0_time = df_0_time.loc[(df_0_time >= f'{start_time}') & (df_0_time <= f'{end_time}')]
    df_0_time_array = df_0_time.to_numpy()  # 得到原始推文中的全部时间列表

    arr_time = []
    arr_count = []
    for time_0 in df_0_time_array:
        count_0 = df_0["Timestamp"].str.contains(str(time_0)).sum()  # 根据条件进行查询，并统计个数
        time_0 = datetime.strptime(time_0, '%Y-%m-%d')
        arr_time.append(time_0)
        arr_count.append(int(count_0))

    return arr_time, arr_count # 返回每个地区推文中出现的时间和次数

def covert_time_count(time_0, count_0, start_time, end_time):
    time_0 = [str(i.strftime('%Y-%m-%d')) for i in time_0]  # 将datetime转换成str类型的时间
    time_0, count_0 = no_data_add_0(time_0, count_0, start_time, end_time) # 添加新数据
    time_0, count_0 = adjust_time_correct_order(time_0, count_0)  # 重新调整顺序
    return time_0, count_0

def min_max_normalization(input_array):
    input_array = np.array(input_array)
    min_value = np.min(input_array)
    # print(min_value)
    max_value = np.max(input_array)
    # print(max_value)
    normalized_array = 2 * (input_array - min_value) / (max_value - min_value) - 1
    return normalized_array

def get_time_array(start_time, end_time):
    file_dir = r"F:\Python_Files\Python_Project_02\Twitter_scrape\Twitter_scrape\xlsx_process_0809\time_series_clustering\region_xlsx"
    xlsx_0 = "Northern Africa.xlsx"
    xlsx_1 = "Sub-Saharan Africa.xlsx"
    xlsx_2 = "Latin America and the Caribbean.xlsx"
    xlsx_3 = "Northern America.xlsx"
    xlsx_4 = "Eastern Asia.xlsx"
    xlsx_5 = "South-eastern Asia.xlsx"
    xlsx_6 = "Southern Asia.xlsx"
    xlsx_7 = "Western Asia.xlsx"
    xlsx_8 = "Eastern Europe.xlsx"
    xlsx_9 = "Northern Europe.xlsx"
    xlsx_10 = "Southern Europe.xlsx"
    xlsx_11 = "Western Europe.xlsx"
    xlsx_12 = "Australia and New Zealand.xlsx"
    df0 = pd.read_excel(file_dir + "\\" + xlsx_0)
    df1 = pd.read_excel(file_dir + "\\" + xlsx_1)
    df2 = pd.read_excel(file_dir + "\\" + xlsx_2)
    df3 = pd.read_excel(file_dir + "\\" + xlsx_3)
    df4 = pd.read_excel(file_dir + "\\" + xlsx_4)
    df5 = pd.read_excel(file_dir + "\\" + xlsx_5)
    df6 = pd.read_excel(file_dir + "\\" + xlsx_6)
    df7 = pd.read_excel(file_dir + "\\" + xlsx_7)
    df8 = pd.read_excel(file_dir + "\\" + xlsx_8)
    df9 = pd.read_excel(file_dir + "\\" + xlsx_9)
    df10 = pd.read_excel(file_dir + "\\" + xlsx_10)
    df11 = pd.read_excel(file_dir + "\\" + xlsx_11)
    df12 = pd.read_excel(file_dir + "\\" + xlsx_12)
    # 提取surprise情感的推文，不区分SIR阶段
    df0 = df0[df0["prediction"] == "surprise"]  # & (df0["id"] != str(0)) & (df0["Comments"].isnull())
    df1 = df1[df1["prediction"] == "surprise"]
    df2 = df2[df2["prediction"] == "surprise"]
    df3 = df3[df3["prediction"] == "surprise"]
    df4 = df4[df4["prediction"] == "surprise"]
    df5 = df5[df5["prediction"] == "surprise"]
    df6 = df6[df6["prediction"] == "surprise"]
    df7 = df7[df7["prediction"] == "surprise"]
    df8 = df8[df8["prediction"] == "surprise"]
    df9 = df9[df9["prediction"] == "surprise"]
    df10 = df10[df10["prediction"] == "surprise"]
    df11 = df11[df11["prediction"] == "surprise"]
    df12 = df12[df12["prediction"] == "surprise"]
    # 计算数量
    time_0, count_0 = get_time_count(df0, start_time, end_time)
    time_1, count_1 = get_time_count(df1, start_time, end_time)
    time_2, count_2 = get_time_count(df2, start_time, end_time)
    time_3, count_3 = get_time_count(df3, start_time, end_time)
    time_4, count_4 = get_time_count(df4, start_time, end_time)
    time_5, count_5 = get_time_count(df5, start_time, end_time)
    time_6, count_6 = get_time_count(df6, start_time, end_time)
    time_7, count_7 = get_time_count(df7, start_time, end_time)
    time_8, count_8 = get_time_count(df8, start_time, end_time)
    time_9, count_9 = get_time_count(df9, start_time, end_time)
    time_10, count_10 = get_time_count(df10, start_time, end_time)
    time_11, count_11 = get_time_count(df11, start_time, end_time)
    time_12, count_12 = get_time_count(df12, start_time, end_time)
    # 时间格式调整
    time_0, count_0 = covert_time_count(time_0, count_0, start_time, end_time)
    time_1, count_1 = covert_time_count(time_1, count_1, start_time, end_time)
    time_2, count_2 = covert_time_count(time_2, count_2, start_time, end_time)
    time_3, count_3 = covert_time_count(time_3, count_3, start_time, end_time)
    time_4, count_4 = covert_time_count(time_4, count_4, start_time, end_time)
    time_5, count_5 = covert_time_count(time_5, count_5, start_time, end_time)
    time_6, count_6 = covert_time_count(time_6, count_6, start_time, end_time)
    time_7, count_7 = covert_time_count(time_7, count_7, start_time, end_time)
    time_8, count_8 = covert_time_count(time_8, count_8, start_time, end_time)
    time_9, count_9 = covert_time_count(time_9, count_9, start_time, end_time)
    time_10, count_10 = covert_time_count(time_10, count_10, start_time, end_time)
    time_11, count_11 = covert_time_count(time_11, count_11, start_time, end_time)
    time_12, count_12 = covert_time_count(time_12, count_12, start_time, end_time)
    # result_time_arr = [count_0, count_1, count_2, count_3, count_4,
    #                    count_5, count_6, count_7, count_8, count_9,
    #                    count_10, count_11, count_12]
    # print(result_time_arr)
    # print("——————————————————————————————————————————————————————————————————————")
    # 时间序列数据进行一阶差分
    count_0 = np.diff(count_0)
    count_1 = np.diff(count_1)
    count_2 = np.diff(count_2)
    count_3 = np.diff(count_3)
    count_4 = np.diff(count_4)
    count_5 = np.diff(count_5)
    count_6 = np.diff(count_6)
    count_7 = np.diff(count_7)
    count_8 = np.diff(count_8)
    count_9 = np.diff(count_9)
    count_10 = np.diff(count_10)
    count_11 = np.diff(count_11)
    count_12 = np.diff(count_12)
    # 数据聚合
    result_time_arr = [count_0, count_1, count_2, count_3, count_4,
                       count_5, count_6, count_7, count_8, count_9,
                       count_10, count_11, count_12]
    # print(result_time_arr)
    # print("——————————————————————————————————————————————————————————————————————")
    # 一阶差分后数据归一化，数据不需要标准化，相关指标数据已经标准化为百分比数据。
    result_time_arr = min_max_normalization(result_time_arr)
    # print(result_time_arr)

    return result_time_arr

def adf_and_kpss(input_array):
    from statsmodels.tsa.stattools import adfuller, kpss
    # 转换成 DataFrame
    df = pd.DataFrame({'Time': input_array})
    # ADF 检验
    adf_result = adfuller(df['Time'])
    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    # 判断 ADF 检验结果
    if adf_result[1] < 0.05:
        print("ADF Test: The time series is likely stationary.")
    else:
        print("ADF Test: The time series is likely non-stationary.")
    # KPSS 检验
    kpss_result = kpss(df['Time'])
    print("KPSS Statistic:", kpss_result[0])
    print("p-value:", kpss_result[1])
    # 判断 KPSS 检验结果
    if kpss_result[1] < 0.05:
        print("KPSS Test: The time series is likely non-stationary.")
    else:
        print("KPSS Test: The time series is likely stationary.")

def contact_time_series_and_index(result_time_arr):
    index_df = pd.read_excel("./193countries_emotion_index_delete_no_tweets.xlsx", sheet_name="Sheet3")
    index_df = index_df[:13]
    # print(index_df.loc[0,"region"])

    result_arr = []

    for i in range(0, 13):
        time_series = result_time_arr[i]
        time_series_length = len(result_time_arr[i])
        # 根据时间序列数组长度生成指标数据数组
        population_percentage = index_df.loc[i, "population_percentage"]
        population_arr = np.full(time_series_length, population_percentage)
        religion_percentage = index_df.loc[i, "religion_percentage"]
        religion_arr = np.full(time_series_length, religion_percentage)
        GDP_percentage = index_df.loc[i, "GDP_percentage"]
        GDP_arr = np.full(time_series_length, GDP_percentage)
        education = index_df.loc[i, "education"]
        education_arr = np.full(time_series_length, education)
        Internet = index_df.loc[i, "Internet"]
        Internet_arr = np.full(time_series_length, Internet)
        # 合并数据
        merge_arr = [time_series, population_arr, religion_arr, GDP_arr, education_arr, Internet_arr]
        # 添加到新数组中
        result_arr.append(merge_arr)

    return result_arr


############################################# K-Means clustering #############################################
def elbow_method(time_series_data, max_cluster, seed, metric):
    # 初始化TimeSeriesScalerMeanVariance
    scaler = TimeSeriesScalerMeanVariance()
    # 对时间序列数据进行标准化处理
    X = scaler.fit_transform(time_series_data)
    X = X.reshape((X.shape[0], -1))
    # 计算不同聚类数下的 WCSS
    SSE = []  # 誤差平方和（sum of the squared errors, SSE）
    max_clusters = max_cluster
    for i in range(1, max_clusters + 1):
        kmeans = TimeSeriesKMeans(n_clusters=i, random_state=seed, metric=metric)
        kmeans.fit(X)
        SSE.append(kmeans.inertia_)
    # 绘制手肘图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), SSE, marker='o', linestyle='--')
    print("kmeans+Elbow Method", SSE)
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    plt.savefig('kmeans+dtw+Elbow Method.jpeg', dpi=600)
    plt.show()
def silhouette_coefficient(time_series_data, max_clusters, seed, metric):
    # 初始化TimeSeriesScalerMeanVariance
    scaler = TimeSeriesScalerMeanVariance()
    # 对时间序列数据进行标准化处理
    X = scaler.fit_transform(time_series_data)
    X = X.reshape((X.shape[0], -1))
    # 计算不同聚类数下的轮廓系数
    silhouette_scores = []
    # max_clusters = 12
    for n_clusters in range(2, max_clusters + 1):
        km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=seed)
        labels = km.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels, metric=metric)
        silhouette_scores.append(silhouette_avg)
    # 绘制轮廓系数图
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    print("silhouette_scores", silhouette_scores)
    plt.title('Silhouette Scores For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig('kmeans+dtw+silhouette_scores.jpeg', dpi=600)
    plt.show()
def kmeans(time_series_data, k, seed, metric):
    # 初始化TimeSeriesScalerMeanVariance
    scaler = TimeSeriesScalerMeanVariance()
    # 对时间序列数据进行标准化处理
    X = scaler.fit_transform(time_series_data)
    X = X.reshape((X.shape[0], -1))
    # 根据手肘图选择最佳聚类数，例如选择3
    n_clusters = k
    # 初始化TimeSeriesKMeans算法
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, verbose=True, random_state=seed)
    # 进行聚类
    labels = km.fit_predict(X)
    # labels = km.labels_
    print("Cluster labels:", labels)
    cluster_centers = km.cluster_centers_

    # 绘制每个聚类的时间序列图
    plt.figure(figsize=(10, 8))
    for cluster in range(n_clusters):
        plt.subplot(n_clusters, 1, cluster + 1)
        for series in X[labels == cluster]:
            plt.plot(series, color='gray', alpha=0.5)
        # plt.plot(cluster_centers[cluster], color='#D10363', lw=2)
        plt.plot(cluster_centers[cluster], color='#D10363', lw=2)
        plt.title(f'Cluster {cluster}')
    plt.tight_layout()
    plt.show()

    # # 绘制聚类中心的时间序列图
    # plt.figure(figsize=(10, 8))
    # for cluster in range(n_clusters):
    #     plt.plot(cluster_centers[cluster], label=f'Cluster {cluster}')
    # plt.title('Cluster Centers Time Series')
    # plt.xlabel('Time Step')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()
############################################# K-Means clustering #############################################

if __name__ == '__main__':

    # 获取时间序列数据，并与指标数据进行聚合
    result_time_arr = np.array(get_time_array("2023-10-07", "2024-05-08")) # 214
    result_time_arr = contact_time_series_and_index(result_time_arr)

    # 使用手肘法判断聚类个数
    # elbow_method(result_time_arr,12,123, "euclidean")
    # 使用轮廓系数
    # silhouette_coefficient(result_time_arr,12,123, "euclidean")
    # 通过手肘法和轮廓系数法判断，聚类个数为3 最佳
    # kmeans(result_time_arr, 4, 123,"euclidean")

    # 使用手肘法判断聚类个数
    elbow_method(result_time_arr, 12, 123, "dtw")
    # 使用轮廓系数
    silhouette_coefficient(result_time_arr, 12, 123, "dtw")
    # # 通过手肘法和轮廓系数法判断，聚类个数为4 最佳
    # kmeans(result_time_arr, 4, 123,"dtw")

    # # 使用手肘法判断聚类个数
    # elbow_method(result_time_arr,12,123, "softdtw")
    # # 使用轮廓系数
    # silhouette_coefficient(result_time_arr,12,123, "softdtw")
    # # 通过手肘法和轮廓系数法判断，聚类个数为3 最佳
    # # kmeans(result_time_arr,3, 123,"softdtw")




