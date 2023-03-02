import random

import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import requests
from bs4 import BeautifulSoup

import time


def to_get_proxies():
    url = "https://free-proxy-list.net/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    ips = soup.select("table.table.table-striped.table-bordered tbody tr td")
    ips = [ips[8 * i:8 * (i + 1)] for i in range(int(len(ips) / 8 + 1))]
    for row in ips:
        if row[6].getText() != "yes":
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


PROXY = to_get_proxies()

CHAR_CHECK = "abcdefghijklmnopqrstuvwxyz"

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--proxy-server=%s' % PROXY["https"])
options = Options()
options.headless = False

DRIVER_PATH = r"C:\Users\dsm84762\Desktop\chromedriver_win32\chromedriver.exe"
driver = webdriver.Chrome(executable_path=DRIVER_PATH, options=options, chrome_options=chrome_options)
driver.get("https://www.npr.org/sections/politics")

while True:
    divs = driver.find_elements(By.CSS_SELECTOR, "div.item-info h2.title a")[-25:]
    article_links = [link.get_attribute("href") for link in divs]

    for link in article_links:
        try:
            request = requests.get(link, proxies=PROXY)
        except requests.exceptions.ProxyError:
            continue
        soup = BeautifulSoup(request.content)
        article_text = ' '.join([paragraph.getText() for paragraph in soup.select("#storytext > p")])
        title = ''.join([letter for letter in article_text[-12:].lower() if letter in CHAR_CHECK]) + str(random.randint(
            1000, 100000)) + '.txt'
        with open(os.path.join(os.getcwd(), 'npr', title), 'w', errors='ignore', encoding='utf-8') as f:
            f.write(article_text)
    # Scroll down, hit next button
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    next_button = driver.find_element(By.CSS_SELECTOR, "button.options__load-more")
    next_button.click()
    time.sleep(4)
