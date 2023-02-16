from bs4 import BeautifulSoup
import urllib.request

import os
import time
import re

# Function used is from ActiveState
# https://code.activestate.com/recipes/81330-single-pass-multiple-replace/
def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


class Scraper:
    def __init__(self):
        self.pages_accessed = 0

    def fox_from_link(self, link: str) -> str: 
        page = urllib.request.urlopen(link)
        soup = BeautifulSoup(page, "html.parser")
        paragraphs = soup.select('.article-body p')
        body_text = ""
        for p in paragraphs:
            body_text += p.getText() + "\n"
        return body_text   

    def slate_from_link(self, link: str) -> str: 
        page = urllib.request.urlopen(link)
        soup = BeautifulSoup(page, "html.parser")
        paragraphs = soup.select(".slate-paragraph.slate-graf")
        body_text = ""
        for p in paragraphs:
            body_text += p.getText()
        return body_text

    def slate_n_articles(self, n: int) -> list:
        page = urllib.request.urlopen("https://slate.com/news-and-politics")
        soup = BeautifulSoup(page, "html.parser")
        texts = []
        article_num = page_num = 0
        while article_num < n:
            links = soup.select("a.topic-story")
            for link in (page['href'] for page in links):
                texts.append(self.article_from_link(link))
                article_num += 1
                if article_num >= n:
                    break
            page = urllib.request.urlopen(f"https://slate.com/news-and-politics/{page_num}#recent")
            soup = BeautifulSoup(page, "html.parser")

        return texts


            
    def msnbc_from_link(self, link: str) -> str:
        page = urllib.request.urlopen(link) 
        soup = BeautifulSoup(page, "html.parser")
        paragraphs = soup.select(".showblog-body__content p")
        full_text = ""
        for p in paragraphs:
            full_text += p.getText()
        return full_text

    def msnbc_n_articles(self, n: int) -> list:
        page = urllib.request.urlopen("https://www.msnbc.com/")
        soup = BeautifulSoup(page, "html.parser")
        link = soup.select(".smorgasbord-meta-content__headline.smorgasbord-meta-content__headline--L a")[0]['href']
        texts = []

        for _ in range(n):
            page = urllib.request.urlopen(link)
            soup = BeautifulSoup(page, features="html.parser")
            texts.append(self.article_from_link(link))

            div = soup.select(".styles_navItemLink__rw4pC")
            next_button = div[0].find('a')
            link = next_button['href']
        return texts

    
    def newsmax_from_link(self, link: str) -> str:
        page = urllib.request.urlopen(link)
        soup = BeautifulSoup(page, "html.parser")
        paragraphs = soup.select("div #mainArticleDiv")
        full_text = ""
        for p in paragraphs:
            full_text += p.getText()
        return full_text

    def newsmax_n_articles(self, n: int) -> list:
        page = urllib.request.urlopen("https://www.newsmax.com/archives/politics/1/2022/12/")
        soup = BeautifulSoup(page, "html.parser")
        texts = []
        month = 12
        year = 2022
        article_num = 0
        while article_num < n:
            links = soup.select("ul.archiveRepeaterUL li.archiveRepeaterLI h5.archiveH5 a")
            for link in ("https://www.newsmax.com" + l['href'] for l in links):
                try:
                    texts.append(self.article_from_link(link))
                except urllib.error.URLError:
                    return texts
                article_num += 1
                if article_num >= n:
                    break
            month -= 1
            if month == 0:
                month = 12
                year -= 1
            page = urllib.request.urlopen(f"https://www.newsmax.com/archives/politics/1/{year}/{month}/")
            soup = BeautifulSoup(page, "html.parser")
        return texts

    def breitbart_from_link(self, link: str) -> str:
        page = urllib.request.urlopen(link)
        soup = BeautifulSoup(page, "html.parser")
        full_text = ""

        paragraphs = soup.select("div.entry-content p")
        for p in paragraphs:
            full_text += p.getText()
        return full_text

    def breitbart_n_articles(self, n: int) -> list:
        page = urllib.request.urlopen("https://www.breitbart.com/politics/")
        soup = BeautifulSoup(page, "html.parser")

        page_num = 1
        article_num = 0
        texts = []

        while article_num < n:
            articles = soup.select("section.aList.cf_show_classic article div h2 a")
            for link in (article['href'] for article in articles):
                try:
                    texts.append(self.article_from_link("https://www.breitbart.com" + link))
                except urllib.error.URLError:
                    return texts
                article_num += 1
                if article_num >= n:
                    break
            page_num += 1
            page = urllib.request.urlopen(f"https://www.breitbart.com/politics/page/{page_num}")
            soup = BeautifulSoup(page, "html.parser")
        return texts


    def article_from_link(self, link):
        site = link.replace("https://", "")
        site = site.replace("www.", "")
        site = site[:3]
        function_dict = {
            "msn" : self.msnbc_from_link,
            "fox" : self.fox_from_link,
            "new": self.newsmax_from_link,
            "sla": self.slate_from_link,
            "bre": self.breitbart_from_link
        }

        self.pages_accessed += 1
        try:
            text = function_dict[site](link)
        except urllib.error.URLError:
            time.sleep(30)
            text = function_dict[site](link)
        return text

    def n_articles(self, link):
        site = link.rsplit("www.", 1)[1][:3]
        function_dict = {
            "msn": self.msnbc_n_articles
        }

        return function_dict[site](link)

    def write_article(self, link, folder, name):
        os.mkdir(folder)
        with open(os.path.join(os.getcwd(), folder, link.rsplit(".com/", 1)[1][:8]), 'w') as f:
            f.write(self.article_from_link(link))

    def create_dataset(self, folders: list, websites: list, n: int) -> None:
        function_dict = {
            "msn": self.msnbc_n_articles,
            "new": self.newsmax_n_articles,
            "sla": self.slate_n_articles,
            "bre": self.breitbart_n_articles
        }
        replace_dict = {
            "\\": "",
            "\n": "",
            "\b": "",
            "!": "",
            " ": "",
            "/": "",
            ".": "",
            "_": "",
            "-": "",
            "*": "",
            '"': "",
            "[": "",
            "]": "",
            ":": "",
            ";": "",
            "|": "",
            ",": ""
        }

        for folder, website in zip(folders, websites):
            cnt = 1
            directory = os.path.join(os.getcwd(), folder)
            if not os.path.exists(directory):
                os.makedirs(directory)
            func = function_dict[website]
            articles = func(n)
            for article in articles:
                title = multiple_replace(replace_dict, article[:10]) + str(cnt) + ".txt"
                with open(os.path.join(directory, title), 'w', errors='ignore', encoding='utf-8') as f:
                    f.write(article)
                cnt += 1
    

if __name__ == "__main__":
    test = Scraper()
    test.create_dataset(['newsmax', 'breitbart', 'slate'], ['new', 'bre', 'sla'], 50)
