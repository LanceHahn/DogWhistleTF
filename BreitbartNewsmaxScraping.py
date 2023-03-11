import requests
from bs4 import BeautifulSoup

from NPRScraping import get_proxy

import random
import os


def newsmax_n_articles(n: int) -> list:
    proxy = get_proxy()
    page = requests.get('https://www.newsmax.com/archives/politics/1/2022/7/')
    soup = BeautifulSoup(page.content, "html.parser")
    month = 7
    year = 2022
    article_num = 0
    while article_num < n:

        links = soup.select("ul.archiveRepeaterUL li.archiveRepeaterLI h5.archiveH5 a")
        for link in ("https://www.newsmax.com" + l['href'] for l in links):
            try:
                article_page = requests.get(link, verify=False)
            except:
                try:
                    article_page = requests.get(link, proxies=proxy, verify=False)
                except:
                    proxy = get_proxy()
                    article_page = requests.get(link, proxies=proxy, verify=False)
            soup = BeautifulSoup(article_page.content, "html.parser")
            full_text = ' '.join([paragraph.getText() for paragraph in soup.select("div #mainArticleDiv")])
            yield full_text
            article_num += 1
            if article_num >= n:
                break
        month -= 1
        if month == 0:
            month = 12
            year -= 1
        page = requests.get(f"https://www.newsmax.com/archives/politics/1/{year}/{month}/")
        soup = BeautifulSoup(page.content, "html.parser")


def breitbart_n_articles(n: int, page_num=1) -> list:
    page = requests.get(f"https://www.breitbart.com/politics/page/{page_num}")
    soup = BeautifulSoup(page.content, "html.parser")
    article_num = 0

    while article_num < n:
        articles = soup.select("section.aList.cf_show_classic article div h2 a")
        for link in (article['href'] for article in articles):
            try:
                article_page = requests.get("https://www.breitbart.com" + link, verify=False)
            except:
                proxy = get_proxy()
                article_page = requests.get("https://www.breitbart.com" + link, proxies=proxy, verify=False)
            soup = BeautifulSoup(article_page.content, "html.parser")
            full_text = ' '.join([p.getText() for p in soup.select("div.entry-content p")])
            yield full_text
            article_num += 1
            if article_num >= n:
                break
        page_num += 1
        page = requests.get(f"https://www.breitbart.com/politics/page/{page_num}")
        soup = BeautifulSoup(page.content, "html.parser")


def npr_n_articles(n):
    proxy = get_proxy()
    cnt = 0
    with open("nprlinks.txt", 'r') as f:
        for _ in range(5185):
            f.readline()
        while True:
            cnt += 1
            next_link = f.readline()
            if not next_link or cnt > n:
                break
            try:
                page = requests.get(next_link, verify=False)
            except:
                try:
                    page = requests.get(next_link, proxies=proxy, verify=False)
                except:
                    proxy = get_proxy()
                    page = requests.get(next_link, proxies=proxy, verify=False)


            soup = BeautifulSoup(page.content, "html.parser")
            full_text = ' '.join([paragraph.getText() for paragraph in soup.select("#storytext > p")])
            yield full_text


if __name__ == "__main__":
    CHAR_CHECK = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789-_:;."
    fetch_newsmax = newsmax_n_articles(50_000)
    fetch_npr = npr_n_articles(50_000)
    while True:

        npr = next(fetch_npr)
        newsmax = next(fetch_newsmax)

        npr_title = ''.join([letter for letter in'NPR-' + npr[10:18] + '-' + str(random.randint(1000, 9999)) + '.txt' if letter in CHAR_CHECK])
        newsm_title = ''.join([letter for letter in 'NEW-' + newsmax[10:18] + '-' + str(random.randint(1000, 9999)) + '.txt' if letter in CHAR_CHECK])

        with open(os.path.join("npr_articles", npr_title), 'w', errors='ignore', encoding='utf-8') as npr_f, open(os.path.join("newsmax_articles", newsm_title), 'w', errors='ignore', encoding='utf-8') as newsmax_f:
            npr_f.write(npr)
            newsmax_f.write(newsmax)


