import urllib.request
from bs4 import BeautifulSoup

import os
import random

if __name__ == "__main__":
    breitbart_pagen = slate_pagen = 1
    breitbart_articlen = slate_articlen = 1
    title_check = "abcdefghijklmnopqrstuvwxyz"

    while True:
        pass
        # Fetch 1 page of Breitbart articles as BeautifulSoup objects
        try:
            breit_politics = urllib.request.urlopen(f"https://www.breitbart.com/politics/page/{breitbart_pagen}")
            breit_politics = BeautifulSoup(breit_politics)
            breit_links = ["https://www.breitbart.com" + link['href'] for link in breit_politics.select("section.aList.cf_show_classic article div h2 a")]
            breit_articles = [urllib.request.urlopen(link) for link in breit_links]
            breit_articles = [BeautifulSoup(article) for article in breit_articles]
            for article in breit_articles:
                article = ' '.join([paragraph.getText() for paragraph in article.select("div.entry-content p")])
                title = ''.join([letter for letter in article[-12:].lower() if letter in title_check]) + str(breitbart_articlen) + '.txt'
                if len(title) < 8:
                    title = ''.join(random.sample(title_check, 8)) + title
                path = os.path.join(os.getcwd(), "breitbart_articles", title)
                with open(path, 'w', errors='ignore', encoding='utf-8') as f:
                    f.write(article)
                    breitbart_articlen += 1
            breitbart_pagen += 1


        except:
            pass
        # Fetch 1 page of Slate articles as BeautifulSoup objects
        try:
            slate_politics = urllib.request.urlopen(f"https://slate.com/news-and-politics/{slate_pagen}#recent")
            slate_politics = BeautifulSoup(slate_politics)
            slate_links = [a['href'] for a in slate_politics.select("a.topic-story")]
            slate_articles = [urllib.request.urlopen(link) for link in slate_links]
            slate_articles = [BeautifulSoup(article) for article in slate_articles]
            for article in slate_articles:
                article = ' '.join([paragraph.getText() for paragraph in article.select(".slate-paragraph.slate-graf")])
                title = ''.join([letter for letter in article[-12:].lower() if letter in title_check]) + str(slate_articlen) + '.txt'
                if len(title) < 8:
                    title = ''.join(random.sample(title_check, 8)) + title
                path = os.path.join(os.getcwd(), "slate_articles", title)
                with open(path, 'w', errors='ignore', encoding='utf-8') as f:
                    f.write(article)
                    slate_articlen += 1
            slate_pagen += 1
        except:
            pass

