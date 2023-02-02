from bs4 import BeautifulSoup
import urllib.request

import os

class Scraper:
    def __init__(self):
        self.pages_accessed = 0

    def msnbc_from_link(self, link: str) -> str:
        page = urllib.request.urlopen(link) 
        soup = BeautifulSoup(page, "lxml")
        paragraphs = soup.select(".showblog-body__content p")
        full_text = ""
        for p in paragraphs:
            full_text += p.getText()
        return full_text

    def fox_from_link(self, link: str) -> str: 
        page = urllib.request.urlopen(link)
        soup = BeautifulSoup(page, "html.parser")
        paragraphs = soup.select('.article-body p')
        body_text = ""
        for p in paragraphs:
            body_text += p.getText() + "\n"
        return body_text   

    def msnbc_n_articles(self, link: str, n: int) -> list:
        texts = []
        text = ""
        for i in range(n):
            page = urllib.request.urlopen(link)
            self.pages_accessed += 1
            soup = BeautifulSoup(page, features="lxml")
            texts.append(self.article_from_link(link))

            div = soup.select(".styles_navItemLink__rw4pC")
            next_button = div[0].find('a')
            link = next_button['href']
        return texts

    def article_from_link(self, link):
        site = link.rsplit("www.", 1)[1][:3]
        
        function_dict = {
            "msn" : self.msnbc_from_link,
            "fox" : self.fox_from_link 
        }

        self.pages_accessed += 1

        return function_dict[site](link)
    
    def n_articles(self, link):
        site = link.rsplit("www.", 1)[1][:3]

        function_dict = {
            "msn": self.msnbc_n_articles
        }

    def write_article(self, link, folder, name):
        os.mkdir(folder)
        with open(os.path.join(os.getcwd(), folder, link.rsplit(".com/", 1)[1][:5] ), 'w') as f:
            f.write(self.article_from_link(link))

