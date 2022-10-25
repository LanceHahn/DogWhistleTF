import urllib.request
from bs4 import BeautifulSoup
from PresidentialScraper import extract_speech
from preTrainedSample import organizeData, loadEmbedding, designModel, trainModel, useModel
import os
import io

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
                samples.append(extract_speech(
                    urllib.request.urlopen("https://www.presidency.ucsb.edu" + article.select('a')[1]['href'])))
                classes.append(label)
            try:
                page = urllib.request.urlopen("https://www.presidency.ucsb.edu" +
                                              soup.find('a', {'title': 'Go to next page'})['href'])
                soup = BeautifulSoup(page, "html.parser")
            except TypeError:
                break
    return classes, samples


def writeData(labels, searches, new_folder_title):
    """
    Write given speeches to a specified directory.

    :param labels: Ordered list of the labels matching presidential archive links
    :type labels: list
    :param searches: Ordered list of presidential archive links, which will be scraped and assigned labels
    :type searches: list
    :param new_folder_title: Title of folder where you want to store the files
    :type new_folder_title: str
    :return: No return value
    """
    folder_path = os.path.join(os.getcwd(), new_folder_title)
    os.mkdir(folder_path)

    page_count = 0
    for label, search in zip(labels, searches):
        current_path = os.path.join(folder_path, f"{label}")
        os.mkdir(current_path)
        art_num = 0

        search_page = urllib.request.urlopen(search)
        soup = BeautifulSoup(search_page, "html.parser")
        while search_page is not None and page_count < 2:
            print(f"{page_count * 100} pages out of"
                  f"{ soup.select_one('div.view-header').getText().split('of')[1].split('records')[0]} processed")
            page_count += 1
            evens = soup.select(".even")
            odds = soup.select(".odd")
            for i in odds:
                evens.append(i)
            with io.open(os.path.join(current_path, f"{label}_sample_{art_num}.txt"), 'w', encoding="utf-8"):
                for article in evens:
                    # print(f"Processing article #{art_num} of"
                    #     f"{ soup.select_one('div.view-header').getText().split('of')[1].split('records')[0]}")
                    f = open(os.path.join(current_path, f"{label}_sample_{art_num}.txt"), 'w', errors="ignore")
                    f.write(extract_speech(
                        urllib.request.urlopen("https://www.presidency.ucsb.edu" + article.select('a')[1]['href'])))
                    art_num += 1
            try:
                search_page = urllib.request.urlopen("https://www.presidency.ucsb.edu" +
                                                     soup.find('a', {'title': 'Go to next page'})['href'])
                soup = BeautifulSoup(search_page, "html.parser")
            except TypeError:
                break


def fetchData(folder_title):
    """
    Return a list of sample and labels after the writeData function was used for presidential data.

    :param folder_title: Title of the folder containing the scraped data, the same folder as the new_folder_title param in writeData
    :type folder_title: str
    :return: labels, samples
    """
    labels = []
    samples = []
    label_index = 0

    main_dir = os.path.join(os.getcwd(), folder_title)
    class_list = os.listdir(main_dir)
    for label in class_list:
        for file in os.listdir(os.path.join(main_dir, label)):
            labels.append(label_index)
            temp = open(os.path.join(main_dir, label, file), 'r', errors='ignore', encoding='utf-8')
            samples.append(temp.read())
            temp.close()
        label_index += 1
    return labels, samples


if __name__ == "__main__":

    labels = ["trump", "obama"]
    searches = ["https://www.presidency.ucsb.edu/advanced-search?field-keywords=&field-keywords2=&field-keywords3=&from%5Bdate%5D=10-10-2018&to%5Bdate%5D=07-06-2019&person2=200301&category2%5B%5D=&items_per_page=100",
                "https://www.presidency.ucsb.edu/advanced-search?field-keywords=&field-keywords2=&field-keywords3=&from%5Bdate%5D=10-10-2015&to%5Bdate%5D=07-06-2016&person2=200300&items_per_page=100"]
    writeData(labels, searches, "small_data")
    r_labels, r_samples = fetchData("small_data")
    classLabels = [dirname for dirname in sorted(os.listdir(os.getcwd() + "//small_data"))]
    train_samples, train_labels, val_samples, val_labels, vectorizer = organizeData(r_labels, r_samples, 10, .2)
    embeddingSpace = loadEmbedding(vectorizer, os.path.join(os.getcwd(), "glove.6B", "glove.6B.50d.txt"))
    modelArch = designModel(embeddingSpace, len(classLabels), 10, vectorizer)
    modelTrained = trainModel(train_samples, train_labels, val_samples, val_labels, vectorizer, modelArch)
    useModel(modelTrained, classLabels)
