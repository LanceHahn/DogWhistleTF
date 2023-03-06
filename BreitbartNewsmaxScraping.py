import requests
from bs4 import BeautifulSoup

from NPRScraping import get_proxy

import random
import os

def newsmax_n_articles(n: int) -> list:
    proxy = get_proxy()
    page = requests.get("https://www.newsmax.com/archives/politics/1/2022/12/")
    soup = BeautifulSoup(page.content, "html.parser")
    month = 12
    year = 2022
    article_num = 0
    while article_num < n:
        links = soup.select("ul.archiveRepeaterUL li.archiveRepeaterLI h5.archiveH5 a")
        for link in ("https://www.newsmax.com" + l['href'] for l in links):
            try:
                article_page = requests.get(link, proxies=proxy)
            except:
                proxy = get_proxy()
                article_page = requests.get(link, proxies=proxy)
            soup = BeautifulSoup(article_page, "html.parser")
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
    proxy = get_proxy()
    article_num = 0

    while article_num < n:
        articles = soup.select("section.aList.cf_show_classic article div h2 a")
        for link in (article['href'] for article in articles):
            try:
                article_page = requests.get("https://www.breitbart.com" + link, proxies=proxy)
            except:
                proxy = get_proxy()
                article_page = requests.get("https://www.breitbart.com" + link, proxies=proxy)

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
    cnt = 0
    proxy = get_proxy()
    with open("nprlinks.txt", 'r') as f:
        while True:
            cnt += 1
            next_link = f.readline()
            if not next_link or cnt > n:
                break
            try:
                page = requests.get(next_link, proxies=proxy)
            except:
                proxy = get_proxy()
                page = requests.get(next_link, proxies=proxy)
            soup = BeautifulSoup(page.content)
            full_text = ' '.join([paragraph.getText() for paragraph in soup.select("#storytext > p")])
            yield full_text


if __name__ == "__main__":
    fetch_newsmax = newsmax_n_articles(50_000)
    fetch_breitbart = breitbart_n_articles(50_000)
    fetch_npr = npr_n_articles(50_000)
    while True:
        npr = next(fetch_npr)
        breit = next(fetch_breitbart)
        newsmax = next(fetch_newsmax)
        npr_title = 'NPR-' + npr[10:18] + '-' + str(random.randint(1000, 9999)) + '.txt'
        breit_title = 'BRE-' + breit[10:18] + '-' + str(random.randint(1000, 9999)) + '.txt'
        newsm_title = 'NEW-' + newsmax[10:18] + '-' + str(random.randint(1000, 9999)) + '.txt'
        with open(os.path.join("npr_articles", npr_title), 'w', errors='ignore', encoding='utf-8') as npr_f, open(os.path.join("breit_articles", breit_title), 'w', errors='ignore', encoding='utf-8') as breitbart_f, open(os.path.join("newsmax_articles", newsm_title), 'w', errors='ignore', encoding='utf-8') as newsmax_f:
            npr_f.write(npr)
            breitbart_f.write(breit)
            newsmax_f.write(newsmax)
