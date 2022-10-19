import urllib.request
from bs4 import BeautifulSoup
from PresidentialScraper import extract_speech

import os
import platform


def createData(labels, searches):
    """
    Create a dataset that can later be fed to organizeData(). Better for smaller datasets/testing

    :param labels: Ordered list of the labels to be assigned to the data generated from the links
    :type labels: list
    :param searches: Ordered list of presidential archive links, which will be scraped and assigned labels
    :type searches: list
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


def writeData(labels, searches, new_folder_directory):
    """
    Write given speeches to a specified directory.

    :param labels: Ordered list of the labels to be assigned to the data generated from the links
    :type labels: list
    :param searches: Ordered list of presidential archive links, which will be scraped and assigned labels
    :type searches: list
    :param new_folder_directory: Absolute path of where you want to store the files
    :type new_folder_directory: str
    :return:
    """
    os.mkdir(new_folder_directory)
    samples = []
    classes = []
    page_count = 0
    for label, search in zip(labels, searches):
        current_path = os.path.join(new_folder_directory, f"{label}")
        os.mkdir(current_path)

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
                temp = open(os.path.join(current_path, f"{article.select_one('title')}"), 'w')
                temp.write(label)
                temp.write("\n")
                temp.write(extract_speech(urllib.request.urlopen("https://www.presidency.ucsb.edu" + article.select('a')[1]['href'])))
            try:
                page = urllib.request.urlopen("https://www.presidency.ucsb.edu" +
                                              soup.find('a', {'title': 'Go to next page'})['href'])
                soup = BeautifulSoup(page, "html.parser")
            except TypeError:
                break




