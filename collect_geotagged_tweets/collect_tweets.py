from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers
from Scweet.scweet import scrape
from Scweet.utils import init_driver, log_in
import json
import pandas as pd
import math

def twitter_login():
    ## 打开chrome驱动
    driver = init_driver(headless = False, show_images = False, proxy = None)

    ## 登录账户
    log_in(driver, env = "./env_files/.env", env_id=1)

    return driver


def use_language_to_get_words(language_json, language_str):
    # 加载 JSON 文件
    with open(language_json, 'r', encoding='UTF-8') as f:
        data = json.load(f)
    # 将字符串转换为 JSON 对象
    # print(data)
    words_str = str(data[str(language_str)])
    # print()
    # print("正在运行根据语言的key获取关键词的value程序", words_str)
    # word_0 = words_str.split(",")[0]
    # word_1 = words_str.split(",")[1]
    # word_2 = words_str.split(",")[2]
    # return word_0 + "," +word_1 + "," +word_2
    return words_str

def use_cityName_to_get_capital_geocode(xlsx_path, city_name):
    df = pd.read_excel(xlsx_path, sheet_name='Sheet1')
    df = pd.DataFrame(df)

    country_xlsx_name = ""
    city_gecode = ""
    city_xlsx_name = ""

    for i in range(len(df)):
        city_xlsx_name = str(df.iloc[i][6])
        if city_xlsx_name == city_name:
            country_xlsx_name = str(df.iloc[i][3]) # 此时的国家名称

            ######### 计算城市中心圆半径
            city_area = df.iloc[i][7] # 首都城市面积
            city_radius = float(math.sqrt(float(city_area) / 3.14)) # 计算半径中心
            print("正在计算的城市半径是：{}".format(city_radius))
            ######### 计算城市中心圆半径

            ######### 获取城市中心经纬度
            city_gecode = str(df.iloc[i][8]) + ",{}km".format(city_radius) # 首都中心geocode
            print("正在运行获取国家:{}___首都:{}的geocode的程序".format(country_xlsx_name,
                                                                        city_xlsx_name))
            ######### 获取城市中心圆半径
            break

    return country_xlsx_name, city_xlsx_name, city_gecode



def batch_get_tweets(xlsx_row_num, driver):
    # json路径
    language_json = "./language_json_2.json"
    # xlsx路径
    xlsx_path = "./country_and_city_information.xlsx"

    # 设置相关参数
    # since_time = "2023-10-07" # 已确定
    # untile_time = "2023-10-14" # 已确定
    # language = None #已确定
    # interval = 1 # 已确定
    # geocode = geocode # 根据csv内容获取
    # words = words # 根据csv内容获取
    # save_dir = save_dir # 根据csv内容获取
    # 设置相关参数

    df_xlsx = pd.read_excel(xlsx_path, sheet_name='Sheet1')
    df_xlsx = pd.DataFrame(df_xlsx)

    for j in range(xlsx_row_num, len(df_xlsx)):
        print("序号：{}".format(j))
        language_xlsx_name = str(df_xlsx.iloc[j][4])  # Language
        print("___________________________________________")
        print("正在处理国家:{}__城市{}的数据".format(df_xlsx.iloc[j][0],
                                                     df_xlsx.iloc[j][6]))  # Capital_and_other_cities

        ##### 获取关键词
        words = []
        for language_split_str in language_xlsx_name.split(","):  # 这里修改为粗略查询和精确查询
            keywords = use_language_to_get_words(language_json, language_split_str)
            # words = words + " " + word
            for word in keywords.split(","):
                words.append(word)
        print("正在处理的关键词是:{}".format(words))
        # https://twitter.com/search?q=(%20Isra%C3%ABl%20Palestine%20Israel%20Palestine)%20until%3A2023-11-06%20since%3A2023-10-07%20%20-filter%3Areplies%20geocode%3A45.5410170211942%2C-73.6535336673155%2C11.722643992103704km&src=typed_query&f=live
        # 精确 https://twitter.com/search?f=live&q=%22england%20football%22%20until%3A2012-10-21%20since%3A2007-11-18&src=typed_query
        # 粗略 https://twitter.com/search?f=live&q=(england%20OR%20football)&src=typed_query
        # https://twitter.com/search?q=(Isra%C3%ABl%20OR%20Palestine%20OR%20Israel%20OR%20Palestine)%20until%3A2023-11-06%20since%3A2023-10-07%20%20-filter%3Areplies%20geocode%3A45.5410170211942%2C-73.6535336673155%2C11.722643992103704km&src=typed_query&f=live
        # https://twitter.com/search?q=(Israël%20OR%20Palestine%20OR%20Israel%20OR%20Palestine)%20until%3A2023-11-06%20since%3A2023-10-07%20%20-filter%3Areplies%20geocode%3A45.5410170211942%2C-73.6535336673155%2C11.722643992103704km&src=typed_query&f=live
        ##### 获取关键词

        # print(language_split_str)
        # 开始处理数据
        # language_str = language_split_str # 获取正在处理的语言名称
        # 获取此时的城市名称
        city_name = str(df_xlsx.iloc[j][6])

        # 获取国家首都和geocode信息
        country_str, city_str, geocode_str = use_cityName_to_get_capital_geocode(xlsx_path, city_name)
        # print("geocode_str:", geocode_str)
        # print("geocode_str:", type(geocode_str))

        # 获取对应语言的关键词信息
        # words = use_language_to_get_words(language_json, language_str)
        # print(type(words))

        # 定义输出路径
        save_dir = "outputs_Israel-Gaza_War_0406/" + "{}_{}".format(country_str, city_str)
        print("正在处理国家:{}___首都:{}___Geocode:{}___的数据".format(country_str,
                                                                       city_str,
                                                                       geocode_str))
        # 开始运行程序
        data = scrape(words=words, driver=driver, filter_replies=True, filter_links=True,  # 设置不需要回复推文
                      since="2023-10-07", until="2024-04-04", interval=2,
                      lang=None, from_account=None, headless=False,
                      display_type="Latest", save_images=False,
                      proxy=None, resume=False, save_dir=save_dir,
                      proximity=False, geocode=geocode_str
                      )  # 2023-10-07  # 2024-04-04
        print("已完成国家:{}___首都:{}___Geocode:{}___数据的获取".format(country_str,
                                                                         city_str,
                                                                         geocode_str))
        print("___________________________________________")


if __name__=="__main__":

    # 登录
    driver = twitter_login()

    # 运行主程序
    batch_get_tweets(22, driver)





# https://twitter.com/search?q=(%E6%97%A5%E6%9C%AC%E6%A0%B8%E5%BA%9F%E6%B0%B4)%20until%3A2021-04-17%20since%3A2021-04-16%20&src=typed_query&f=live
# 649435403@qq.com // @wang_10086
# 970113666@qq.com // @wang09183650037
# 1003212442@qq.com // @wangdj156848
# 15621341675@163.com  // @wangdajian68191
# wdj2022nnu@163.com // @dajiangwan66483
# woshiwangdajiang@qq.com  // @dajiangwan2779"
# woshiwangdajiang@gmail.com // @dahaiwang259915




