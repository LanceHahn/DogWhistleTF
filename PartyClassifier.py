import urllib.request
from bs4 import BeautifulSoup
from PresidentialScraper import extract_speech, multiple_replace


def createData(labels, searches):
    """
    Create a dataset that can later be fed to organizeData()

    :param labels: Ordered list of the labels to be assigned to the data generated from the links
    :type labels: list
    :param searches: Ordered list of presidential archive links, which will be scraped and assigned labels
    :return: list of labels, list of texts
    """

    samples = []
    classes = []
    page_count = 0
    for label, search in zip(labels, searches):
        search_page = urllib.request.urlopen(search)
        soup = BeautifulSoup(search_page, "html.parser")
        while search_page is not None:
            print(f"Processing query page {page_count}")
            page_count += 1
            evens = soup.select(".even")
            odds = soup.select(".odd")
            for i in odds:
                evens.append(i)
            for article in evens:
                samples.append(extract_speech(urllib.request.urlopen("https://www.presidency.ucsb.edu" + article.select('a')[1]['href'])))
                classes.append(label)
            try:
                page = urllib.request.urlopen("https://www.presidency.ucsb.edu" +
                                              soup.find('a', {'title': 'Go to next page'})['href'])
                soup = BeautifulSoup(page, "html.parser")
            except TypeError:
                break
    return classes, samples

