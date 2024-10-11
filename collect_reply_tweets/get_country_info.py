"""
模拟登陆访问X.com，查看user用户信息，并返回
"""
import time
from selenium.webdriver.common.by import By
import random
import pandas as pd
import os
from main_utils import log_in_new, create_driver
from Scweet.utils import init_driver, log_in
import re
import numpy as np


def keep_scroling(driver):
    """ scrolling function for tweets crawling"""
    if len(driver.find_elements(by=By.XPATH,
                                value='//script[@data-testid="UserProfileSchema-test"]')) != 0:
        user_page_profile = driver.find_element(by=By.XPATH,
                                                value='//script[@data-testid="UserProfileSchema-test"]').get_attribute('innerHTML')
        print(user_page_profile)
        return user_page_profile
    # 推文不存在
    # class="css-1jxf684 r-bcqeeo r-1ttztb7 r-qvutc0 r-poiln3"
    elif len(driver.find_elements(by=By.XPATH, value='//span[@class="css-1jxf684 r-bcqeeo r-1ttztb7 r-qvutc0 r-poiln3"]')) != 0:
        elements = driver.find_elements(by=By.XPATH, value='//span[@class="css-1jxf684 r-bcqeeo r-1ttztb7 r-qvutc0 r-poiln3"]')
        for i in elements:
            tweet_error = i.text
            if tweet_error == "此账号不存在":
                print(tweet_error)
                raise ZeroDivisionError("此账号不存在！")
            if tweet_error == "账号已被冻结":
                print(tweet_error)
                raise ZeroDivisionError("账号已被冻结！")

            if tweet_error == "出错了。请尝试重新加载。": # # 账号访问频繁，需要更换账号
                print(tweet_error)
                raise IndexError("出错了。请尝试重新加载。")
            if tweet_error == "Something went wrong, but don’t fret — let’s give it another shot.": # # 账号访问频繁，需要更换账号
                print(tweet_error)
                raise IndexError("出错了。请尝试重新加载。")
            if tweet_error == '创建账号':
                raise IndexError("此账号登录频繁，需要切换账号。")
    # 返回账号登录界面，需要更换账号
    elif len(driver.find_elements(by=By.XPATH, value='//input[@autocomplete="username"]')) != 0:
        raise IndexError("出错了。请尝试重新加载。")




class NoTweetError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)  # 初始化父类
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo
class UserAccountError(Exception):
    pass
def get_reply_tweets(url_path, driver, output_file_path):
    driver.get(url_path)
    time.sleep(1)
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    # start scrolling and get tweets
    user_page_profile = keep_scroling(driver)
    # print("已获取位置信息！")
    return user_page_profile


if __name__ == "__main__":

    origin_xlsx = "./123-no-repeat-username.pkl"
    df = pd.read_pickle(origin_xlsx)
    print("总行数: ", len(df))  # 25940

    count = 22917 #
    # 定义计数器 num
    num = 22917 #

    # 账号编号
    account_ids = [7, 8, 9, 10, 11]
    account_index = 0
    account_length = len(account_ids)

    # 开启无限循环
    while True:
        # 根据数字判断账号，目前共有12个账号，0-11，依次遍历，如果报错，切换下一个
        account_index = (account_index) % account_length  # 当 index 达到长度时，它会回到 0
        account_id = account_ids[account_index]
        print("account_id: ", account_id)

        try: # 尝试登录，如果登录界面显示失败，则切换下一个账号
            # 加载驱动
            chromedriver_path = "E:/Anaconda/envs/labelme/chromedriver.exe"
            # 登录
            driver = init_driver(headless=True, show_images=False, proxy=None, chromedriver_path=chromedriver_path)
            # driver.set_window_size(600, 1000) # 设置浏览器窗口大小
            log_in_new(driver, env="./env/.env", rand_id = account_id)
            time.sleep(5)
            # 登录成功后开始获取数据
        except ValueError as e: # 账号登录报错，直接跳出此循环，开始下一循环
            print(f"报错：{e}")
            account_index = account_index + 1 # 账号编号加1
            time.sleep(10)
            continue

        count = num #

        # 开始循环获取数据
        for df_i in range(count, len(df)):
            num = df_i
            if df_i % 100 == 0:
                time.sleep(20)
                # break # 跳出for循环，防止一个账号遍历多次报错

            # 获取用户名和账号名
            user_name = df["UserName_1"][df_i]
            print("xlsx_number:", df_i)
            print("user_name: ", user_name)
            # 保存文件的地址
            output_file_path = "./outputs_get_country_info_4/"
            # url_link = "https://x.com/" + user_name.split("@")[1]
            url_link = "https://x.com/" + user_name
            print("url_link: ", url_link)

            try:  # 获取数据
                # 应该返回一个地理位置的字符串
                user_page_profile = get_reply_tweets(url_path=url_link, driver=driver, output_file_path=output_file_path)
            except ZeroDivisionError as e: # 此账号不存在 或者 账号已被冻结
                print(f"报错：{e}")
                continue
            except IndexError as e:
                print(f"报错：{e}")
                num = num - 5
                time.sleep(10)
                account_index = account_index + 1  # 账号编号加1
                break
            # except ValueError as e:
            #     print(f"报错：{e}")
            #     num = num - 5
            #     time.sleep(10)
            #     break

            # 新建Dataframe，并保存文件
            new_df = pd.DataFrame(columns=["user_profile"])
            new_df.loc[0, 'user_profile'] = user_page_profile
            new_df.to_excel(output_file_path + "/" + user_name + ".xlsx", index=False)
            print("已完成一行: ", df_i)

        # for循环结束的时候，跳出while循环.
        if int(num) == int(len(df) - 1):
            break
    print("任务全部完成！")




