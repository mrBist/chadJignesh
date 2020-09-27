import json
from elasticsearch import Elasticsearch
from tqdm import tqdm
import pandas as pd
import re

INDEX_NAME = "covidnews"
ES_HOST = {"host" : "localhost", "port" : 9200}
NEWS_PATH = "news.csv"


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext


def preprocess_text(string):
    string = cleanhtml(string)
    string = string.replace("\xa0","")


if __name__ == "__main__":
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


    df = pd.read_csv("news.csv")
    df = df[["title", "description","text"]]
    data_dicts = []
    for i in range(len(df)):
        data = df.iloc[i,:]
        data = {"title": data["title"], "description" : data["description"], "text" : preprocess_text(data["text"])}
        data_dicts.append(data)


    ids = 1
    errors = 0
    for data in tqdm(data_dicts):
        try:
            res = es.index(index='covidnews',doc_type='articles',id=ids,body=data)
        except:
            errors += 1
            print(data)
        ids += 1