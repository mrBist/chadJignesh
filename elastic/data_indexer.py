import json
from elasticsearch import Elasticsearch
from matplotlib.pyplot import title
from numpy.lib.utils import source
from tqdm import tqdm
import pandas as pd
import re
import pickle
import os
from tqdm import tqdm
#INDEX_NAME = "covidnews"   have to delete this
import sys
sys.path.append('../')
from build.config import INDEX_NAME,ES_HOST,DOC_TYPE
from build.config import news_article

def save_dict(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)




def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext


def preprocess_text(string):
    string = cleanhtml(string)
    string = string.replace("\xa0","").replace("\n"," ")
    return string


def read_articles(root_dir_path : str):
    articles_pkl = os.listdir(root_dir_path)

    articles : list[news_article] = []
    for article_pkl in tqdm(articles_pkl):
        pkl_path = os.path.join(root_dir_path,article_pkl)
        article = load_dict(pkl_path)

        article = news_article(title = article["title"], text = article["text"],authors = article["authors"],source = article["name"])

        articles.append(article)
    return articles

    


def read_news_csv(news_csv_path):
    df = pd.read_csv(news_csv_path)
    df = df[["title", "description","text", "authors", "url"]]
    articles : list[news_article] = []
    for i in tqdm(range(len(df))):
        data = df.iloc[i,:]
        article = news_article(title= data["title"], text= data["text"],authors= data["authors"], source= data["url"])
        articles.append(article)

    return articles



def convert_news_to_dict(article : news_article ) -> dict:
    return dict(
        authors = str(article.authors),
        text = article.text,
        source = article.source,
        title = article.title
    ) 




def indexer(articles):
    es = Elasticsearch(hosts= [ES_HOST])
    if es.indices.exists(INDEX_NAME):
        print("deleting '%s' index..." % (INDEX_NAME))
        res = es.indices.delete(index = INDEX_NAME)
        print(" response: '%s'" % (res))
    request_body = {
        "settings" : {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }

    print("creating '%s' index..." % (INDEX_NAME))
    res = es.indices.create(index = INDEX_NAME, body = request_body)
    print(" response: '%s'" % (res))


    ids = 1
    errors = 0
    for article in tqdm(articles):
        data = convert_news_to_dict(article= article)
        try:
            res = es.index(index= INDEX_NAME,doc_type= DOC_TYPE,id=ids,body=data)
        except:
            errors += 1
            print(data)
        ids += 1



if __name__ == "__main__":
    itv_pth = os.path.join("../","Resources" ,"scraping_results", "ITV_articles")
    ndtv_pth = os.path.join("../","Resources" ,"scraping_results", "ndtv_articles")
    news_csv_path = os.path.join("../" , "Resources", "scraping_results", "news.csv")
    #articles = (news_csv_path= news_csv_path)
    articles = read_articles(ndtv_pth) + read_articles(itv_pth) + read_news_csv(news_csv_path)
    
    indexer(articles= articles)