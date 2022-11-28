import numpy as np
import pandas as pds

import praw as reddit 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

from transformers import pipeline

api = reddit.Reddit(client_id="uMo68bhbI96WL8eE2I7h_g",
                    client_secret="xqgkfC3ss8_8a6SHC7eTZwUKS3C_XA",
                    user_agent="SecondBot")

def n_top_posts(subreddit_name: str, n: int, post_limit: int = 1000) -> list:
    """
    Extract text body of top n posts from given Subreddit using Python Wrapped Reddit API

    Args:
        subreddit_name (str): Display name of the subreddit
        n (int): Number of posts to return
        post_limit (int, optional): Maximum number of posts to search(this amount of posts will always be fetched, but not searched). Defaults to 1000.

    Returns:
        list: _description_
    """    
    subreddit = api.subreddit(subreddit_name)
    
    print(f"Fetching top {post_limit} posts to search", end="")
    top_posts = subreddit.top(limit=post_limit)
    print("\r", end="")
    print("All posts fetched!                                   ")

    texts = []
    found_count = 0
    for i, post in enumerate(top_posts):
        print(f"\rProcessing post {i+1} of {post_limit}, {found_count} posts with text bodies found", end="")
        text = post.selftext
        if text != "":
            texts.append(text)
            found_count += 1
        if found_count == n:
            print()
            return texts
    print()
    warnings.warn(f"Only {found_count} posts out of the parsed {post_limit} contained text. Returning list of length {found_count}")
    return texts


def n_top_posts_by_keyword(subreddit_name: str, n: int, keywords: list, post_limit: int = 1000) -> list:
    """
    Extract text body of top n posts containing one of the given keywords from given Subreddit using Python Wrapped Reddit API

    Args:
        subreddit_name (str): Display name of the subreddit
        n (int): Number of posts to return
        keywords(list): List containing keywords as strings
        post_limit (int, optional): Maximum number of posts to search(this amount of posts will always be fetched, but not searched). Defaults to 1000.

    Returns:
        list: _description_
    """    
    subreddit = api.subreddit(subreddit_name)
    
    print(f"Fetching top {post_limit} posts to search", end="")
    top_posts = subreddit.top(limit=post_limit)
    print("\r", end="")
    print("All posts fetched!                                   ")

    texts = []
    found_count = 0
    for i, post in enumerate(top_posts):
        print(f"\rProcessing post {i+1} of {post_limit}, {found_count} posts containing keywords found", end="")
        text = post.selftext
        if text != "" and any(word.lower() in text for word in keywords):
            texts.append(text)
            found_count += 1
        if found_count == n:
            print()
            return texts
    print()
    warnings.warn(f"Only {found_count} posts out of the parsed {post_limit} contained keywords. Returning list of length {found_count}")
    return texts


keywords = ["trump", "biden", "racism", "politics", "political", "riot", "protest", "america", "crime", "criminal", "immigrant", "immigration"]
for post in n_top_posts_by_keyword("NoStupidQuestions", 3, keywords, post_limit=5000):
    print(post)