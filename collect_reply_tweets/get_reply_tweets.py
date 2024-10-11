from selenium import webdriver
# from Scweet.utils import init_driver, log_in, get_data
from Scweet.utils import init_driver, log_in
import time
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common import exceptions
import csv
import random
import pandas as pd
import os
from main_utils import log_in_new, create_driver
import re

def get_data(card, save_images=False, save_dir=None):
    """Extract data from tweet card"""
    image_links = []
    try:
        username = card.find_element(by=By.XPATH, value='.//span').text
    except:
        return
    try:
        handle = card.find_element(by=By.XPATH, value='.//span[contains(text(), "@")]').text
    except:
        return
    try:
        postdate = card.find_element(by=By.XPATH, value='.//time').get_attribute('datetime')
    except:
        return

    try:
        text = card.find_element(by=By.XPATH, value='.//div/div/div[2]/div[2]/div[2]').text
    except:
        text = ""
    try:
        embedded = card.find_element(by=By.XPATH, value='.//div/div/div[2]/div[2]/div[3]/div').text
    except:
        embedded = ""

    try:
        reply_cnt = card.find_element(by=By.XPATH, value='.//div[@data-testid="reply"]').text
    except:
        reply_cnt = 0
    try:
        retweet_cnt = card.find_element(by=By.XPATH, value='.//div[@data-testid="retweet"]').text
    except:
        retweet_cnt = 0
    try:
        like_cnt = card.find_element(by=By.XPATH, value='.//div[@data-testid="like"]').text
    except:
        like_cnt = 0
    try:
        elements = card.find_elements(by=By.XPATH, value='.//div[2]/div[2]//img[contains(@src, "https://pbs.twimg.com/")]')
        for element in elements:
            image_links.append(element.get_attribute('src'))
    except:
        image_links = []
    try:
        promoted = card.find_element(by=By.XPATH, value='.//div[2]/div[2]/[last()]//span').text == "Promoted"
    except:
        promoted = False
    if promoted:
        return
    # get a string of all emojis contained in the tweet
    try:
        emoji_tags = card.find_elements(by=By.XPATH, value='.//img[contains(@src, "emoji")]')
    except:
        return
    emoji_list = []
    for tag in emoji_tags:
        try:
            filename = tag.get_attribute('src')
            emoji = chr(int(re.search(r'svg\/([a-z0-9]+)\.svg', filename).group(1), base=16))
        except AttributeError:
            continue
        if emoji:
            emoji_list.append(emoji)
    emojis = ' '.join(emoji_list)
    # tweet url
    try:
        element = card.find_element(by=By.XPATH, value='.//a[contains(@href, "/status/")]')
        tweet_url = element.get_attribute('href')
    except:
        return
    tweet = (
        username, handle, postdate, text, embedded, emojis, reply_cnt, retweet_cnt, like_cnt, image_links, tweet_url)
    return tweet


def keep_scroling(driver, data, writer, tweet_ids, scrolling, tweet_parsed, limit, scroll, last_position, scroll_repeat,
                  save_images=False):
    """ scrolling function for tweets crawling"""

    save_images_dir = "/images"

    if save_images == True:
        if not os.path.exists(save_images_dir):
            os.mkdir(save_images_dir)
    while scrolling and tweet_parsed < limit:
        time.sleep(random.uniform(0.5, 1.5))
        # get the card of tweets
        page_cards = driver.find_elements(by=By.XPATH, value='//article[@data-testid="tweet"]')  # changed div by article
        for card in page_cards:
            tweet = get_data(card, save_images, save_images_dir)
            if tweet:
                # check if the tweet is unique
                tweet_id = ''.join(tweet[:-2])
                if tweet_id not in tweet_ids:
                    tweet_ids.add(tweet_id)
                    data.append(tweet)
                    last_date = str(tweet[2])
                    print("Tweet made at: " + str(last_date) + " is found.")
                    writer.writerow(tweet)
                    tweet_parsed += 1
                    if tweet_parsed >= limit:
                        break # 完全终止当前循环
        scroll_attempt = 0
        while tweet_parsed < limit:
            # check scroll position
            scroll += 1
            print("scroll ", scroll)
            time.sleep(random.uniform(0.5, 1.5))
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);') # 把网页滚动到最底部
            curr_position = driver.execute_script("return window.pageYOffset;")

            # 判断是否到达页面底部
            if last_position == curr_position:

                if len(driver.find_elements(By.XPATH, "//html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div")) != 0:
                    print("存在重试选项——需要切换账号")
                    raise ValueError # return None

                ####### 判断是否卡在初始界面
                elif len(driver.find_elements(By.XPATH, '/html/body/div[1]/div/div/div[1]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[5]/label/div/div[2]/div/input')) != 0:
                    print("卡在初始登录界面")
                    raise ValueError # return None
                ####### 判断是否卡在初始界面

                ####### 到达页面底部后，开始匹配“显示更多回复”按钮，加载更多内容
                elif len(driver.find_elements(By.XPATH, '//span[text()="显示更多回复"]')) != 0:
                    button = driver.find_element(By.XPATH, '//span[text()="显示更多回复"]')
                    print("显示更多回复——选项存在")
                    # button.click() # 点击按钮
                    driver.execute_script("arguments[0].click()", button)
                ####### 到达页面底部后，开始匹配“显示更多回复”按钮，加载更多内容

                else:
                    # print("继续滑动！")
                    # time.sleep(2)
                    scroll_attempt += 1
                    # end of scroll region
                    if scroll_attempt >= 3:
                        scrolling = False
                        scroll_repeat += 1
                        # return
                        break # 完全终止当前循环
                    else:
                        time.sleep(random.uniform(0.5, 1.5))  # attempt another scroll
            else:
                last_position = curr_position
                break
    return driver, data, writer, tweet_ids, scrolling, tweet_parsed, scroll, last_position, scroll_repeat

# def login(chromedriver_path, env):
#     # Chrome浏览器
#     # driver = webdriver.Chrome()
#     ## 定义驱动路径
#     # chromedriver_path="E:/Anaconda/envs/labelme/chromedriver.exe"
#     ## 打开chrome驱动
#     driver = init_driver(headless=False, show_images=False, proxy=None, chromedriver_path=chromedriver_path)
#     ## 登录账户
#     log_in(driver, env)
#     return driver
class NoTweetError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) #初始化父类
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo

def get_reply_tweets(url_path, driver, output_file_path):
    # # 初始化浏览器为chrome浏览器
    # driver = webdriver.Chrome(executable_path=driver)
    # # 设置分辨率 500*500
    # browser.set_window_size(1000,1000)
    # 访问指定网页
    # url_path = r'https://twitter.com/opebanwo/status/1719710439167189048'
    # print("url_path: ", url_path)
    driver.get(url_path)
    time.sleep(1)
    # if len(driver.find_elements(By.XPATH, '/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/div')) != 0:
    if len(driver.find_elements(By.XPATH, '/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div')) != 0:
        # print("此条推文不存在！")
        driver.close()
        raise NoTweetError("此推文不存在！")

    # list that contains all data
    data = []
    # unique tweet ids
    tweet_ids = set()
    # write mode
    write_mode = 'w'
    header = ['UserScreenName', 'UserName', 'Timestamp',
                  'Text', 'Embedded_text', 'Emojis',
                  'Comments', 'Likes', 'Retweets',
                  'Image link', 'Tweet URL']
    limit=float("inf")

    # open the file
    # output_file_path = "./outputs/reply_csv/"
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    file_name = output_file_path + "/" + url_path.split("/")[-1] + ".csv"
    print(file_name)
    with open(file_name, write_mode, newline='', encoding='utf-8') as f:
        print("文件新建成功！")
        writer = csv.writer(f)
        if write_mode == 'w':
            # write the csv header
            writer.writerow(header)
        scroll_repeat = 0
        while scroll_repeat < 1:
            ## 缩进
            scroll = 0
            last_position = driver.execute_script("return window.pageYOffset;")
            scrolling = True
            # number of tweets parsed
            tweet_parsed = 0
            # sleep
            time.sleep(random.uniform(0.5, 1))
            # start scrolling and get tweets
            driver, data, writer, tweet_ids, scrolling, tweet_parsed, scroll, last_position, scroll_repeat = keep_scroling(driver, data, writer,
                                                                                                            tweet_ids, scrolling,
                                                                                                            tweet_parsed, limit, scroll,
                                                                                                            last_position, scroll_repeat)
            ## 缩进
    data = pd.DataFrame(data, columns = ['UserScreenName', 'UserName', 'Timestamp', 'Text', 'Embedded_text',
                                         'Emojis', 'Comments','Likes', 'Retweets','Image link', 'Tweet URL'])
    # close the web driver
    # driver.close()

    # 保存文件
    data.to_excel(output_file_path + "/" + url_path.split("/")[-1] + ".xlsx", index=False)
    print("保存完成！")



if __name__ == "__main__":

    ########### 加载源数据
    origin_xlsx = r"F:\Python_Files\Python_Project_02\Twitter_scrape\Twitter_scrape\outputs_reply\reply_xlsx_4\merge_4_cleaning_translate_llama_final_haveReply.xlsx"
    ########### 加载源数据
    df = pd.read_excel(origin_xlsx, sheet_name='Sheet1', dtype=str)
    print("总行数: ",len(df))  # 2049

    count = 500 # 0
    error_lsit = []
    rand_id = random.randint(0, 6)
    while count < len(df):
    # while count <= 8000:

        try: # 登录
            ##### 创建浏览器驱动并登录
            chromedriver_path = "E:/Anaconda/envs/labelme/chromedriver.exe"
            # 登录
            driver, rand_id, _ = create_driver("./env/", chromedriver_path, rand_id, 7)
            # driver = init_driver(headless=False, show_images=False, proxy=None, chromedriver_path=chromedriver_path)
            # log_in(chromedriver_path, env="./env/.env")
            # time.sleep(2)
            log_in_new(driver, env="./env/.env", rand_id=rand_id)
            # time.sleep(1)
            ##### 创建浏览器驱动并登录
        except Exception:
            print("登录出现异常：", count)
            count = count - 1
            continue


        # 执行数据爬取
        try:
            for df_i in range(count, len(df)):
            # for df_i in range(count, 8001):  #################
                url_link = df["Tweet_URL"][df_i]
                url_id = url_link.split("/")[-1]
                print("xlsx_number:", df_i)
                print("url_link: ", url_link)
                # 保存文件的地址
                output_file_path = "./outputs_reply/reply_xlsx_5"
                # 执行查询
                get_reply_tweets(url_path=url_link, driver=driver, output_file_path=output_file_path)
                print("已完成: ", df_i)
                count += 1
                time.sleep(1)
        except ValueError:
            # print(e)
            print("出现问题: ", count)
            error_lsit.append(count)
            count = count - 1
        except NoTweetError as n:
            print(n)
            count = count + 1  # driver.close()


    else:
        print("全部完成！")
        print("出错列表：", error_lsit)



