from bs4 import BeautifulSoup
import urllib.request

import os

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

    def slate_n_articles(self, n : int) -> list:
        page = urllib.request.urlopen("https://slate.com/")
        soup = BeautifulSoup(page)
        link = soup.select("a.story-card__link")[0]['href']

        texts = []
        for _ in range(n):
            page = urllib.request.urlopen(link)
            soup = BeautifulSoup(page, features="lxml")
            texts.append(self.article_from_link(link))

            next_article = soup.select("a.in-article-recirc__link")[0]
            link = next_article['href']
        return texts


            
    def msnbc_from_link(self, link: str) -> str:
        page = urllib.request.urlopen(link) 
        soup = BeautifulSoup(page, "lxml")
        paragraphs = soup.select(".showblog-body__content p")
        full_text = ""
        for p in paragraphs:
            full_text += p.getText()
        return full_text

    def msnbc_n_articles(self, n: int) -> list:
        page = urllib.request.urlopen("https://www.msnbc.com/")
        soup = BeautifulSoup(page)
        link = soup.select(".smorgasbord-meta-content__headline.smorgasbord-meta-content__headline--L a")[0]['href']
        texts = []

        for _ in range(n):
            page = urllib.request.urlopen(link)
            soup = BeautifulSoup(page, features="lxml")
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
        page = urllib.request.urlopen("https://www.newsmax.com/")
        soup = BeautifulSoup(page)
        link = "https://www.newsmax.com" + soup.select("#nmCanvas6 h1 a.Default")[0]['href']

        texts = []
        for _ in range(n):
            page = urllib.request.urlopen(link)
            soup = BeautifulSoup(page, "html.parser")
            texts.append(self.article_from_link(link))

            next_page_link = soup.select("a.likeCommH2")[0]
            link = "https://www.newsmax.com" + next_page_link['href']
        return texts

    def breitbart_from_link(self, link: str) -> str:
        page = urllib.request.urlopen(link)
        soup = BeautifulSoup(page, "html.parser")
        full_text = ""

        paragraphs = soup.select("p.a8d-pre")
        for p in paragraphs:
            full_text += p.getText()
        return full_text

    def breitbart_n_articles(self, link: str, n: int) -> list:
        texts = []
        for _ in range(n):
            page = urllib.request.urlopen(link)
            soup = BeautifulSoup(page)
            texts.append(self.article_from_link(link))
            
            next_button = soup.select("#DQSW h2 ul li a")[0]
            link = next_button['href']
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

        return function_dict[site](link)

    def n_articles(self, link):
        site = link.rsplit("www.", 1)[1][:3]
        function_dict = {
            "msn": self.msnbc_n_articles
        }

        return function[site](link)
    def write_article(self, link, folder, name):
        os.mkdir(folder)
        with open(os.path.join(os.getcwd(), folder, link.rsplit(".com/", 1)[1][:5] ), 'w') as f:
            f.write(self.article_from_link(link))


if __name__ == "__main__":

    ex = Scraper()
    arts = ex.newsmax_n_articles(2)
    print(arts)