import random

import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import requests
from bs4 import BeautifulSoup

import time


def get_proxy():
    url = "https://free-proxy-list.net/"
    location_check = ["US", "AU", "CA", "FR", "NL", "IN", "JP"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    ips = soup.select("table.table.table-striped.table-bordered tbody tr td")
    ips = [ips[8 * i:8 * (i + 1)] for i in range(int(len(ips) / 8 + 1))]
    for row in ips:
        if row[6].getText() != "yes" or row[2].getText() not in location_check:
            continue
        address = row[0].getText()
        port = row[1].getText()
        proxy = f"http://{address}:{port}"
        proxy_dict = {"http": proxy, "https": proxy}
        try:
            _ = requests.get("https://www.npr.org/sections/politics", proxies=proxy_dict, verify=False)
            return proxy_dict
        except:
            pass
    raise Exception("No working proxies found on free-proxy-list.net")


def fetch_date(date):
    nums = date.split("-")
    year = nums[0]
    month = nums[1]
    day = nums[2]
    month_dict = {
        '2': 31,
        '3': 28,
        '4': 31,
        '5': 30,
        '6': 31,
        '7': 30,
        '8': 31,
        '9': 31,
        '10': 30,
        '11': 31,
        '12': 30
    }

    if month == '1':
        month = 12
        year = int(year) - 1
        day = 31
    elif day == '1':
        day = month_dict[month]
        month = int(month) - 1
    else:
        day = int(day) - 1
    return f"{month}-{day}-{year}"


# PROXY = to_get_proxies()
CHAR_CHECK = "abcdefghijklmnopqrstuvwxyz"

chrome_options = webdriver.ChromeOptions()
options = Options()

DRIVER_PATH = r"C:\Users\dsm84762\Desktop\chromedriver_win32\chromedriver.exe"
driver = webdriver.Chrome(executable_path=DRIVER_PATH, options=options, chrome_options=chrome_options)
url = "https://www.npr.org/sections/politics/archive?date=03-31-2023"
with open("nprlinks.txt", 'w') as f:
    while True:
        driver.get(url)
        last_height = driver.execute_script("return document.body.scrollHeight")

        for _ in range(500):
            links = [article.find_elements(By.CSS_SELECTOR, "div div h2 a")[0] for article in driver.find_elements(By.XPATH, "(/html/body/main/div[2]/div[2]/div[4]/div/article)[position() > last() - 15]")]
            for link in links:
                f.write(link.get_attribute('href') + '\n')
            f.flush()
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            current_height = driver.execute_script("return document.body.scrollHeight")
            if current_height == last_height:
                driver.execute_script("window.scrollBy(0, -250);")
                driver.execute_script("window.scrollBy(0, 350);")
                time.sleep(2)
                current_height = driver.execute_script("return document.body.scrollHeight")
                if current_height == last_height:
                    break
            last_height = current_height
        last_date = driver.find_elements(By.XPATH,"(/html/body/main/div[2]/div[2]/div[4]/div/article)[position() > last() - 1]")[0].find_element(By.CSS_SELECTOR, "time").get_attribute("datetime")
        url = f"https://www.npr.org/sections/politics/archive?date={fetch_date(last_date)}"
'''for link in links:
    try:
        request = requests.get(link, proxies=PROXY)
    except:
        PROXY = to_get_proxies()
        continue
    soup = BeautifulSoup(request.content)
    article_text = ' '.join([paragraph.getText() for paragraph in soup.select("#storytext > p")])
    title = ''.join([letter for letter in article_text[-12:].lower() if letter in CHAR_CHECK]) + str(random.randint(
        1000, 100000)) + '.txt'
    with open(os.path.join(os.getcwd(), 'npr', title), 'w', errors='ignore', encoding='utf-8') as f:
        f.write(article_text)'''

