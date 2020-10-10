import numpy as np
from elasticsearch import Elasticsearch
import sys
sys.path.append("../")
from build.config import INDEX_NAME,ES_HOST, MAX_LEN, TOP_K_ARTICLES, news_article
from model_interface.bertSNLI.bert_nli import BertNLIModel
import json
import os

from typing import List


model_checkpoint_path = os.path.join("../Resources/model/bert-base.state_dict")

es = Elasticsearch(hosts= [ES_HOST])
model_instance = BertNLIModel(model_checkpoint_path)


def convert_dict_to_article(dct : dict, len_chunk = 512) -> news_article:
    dct = dct["_source"]
    convert_to_list = lambda s : s.strip("]").strip("[").split(",")
    article =  news_article(text= dct["text"], title = dct["title"],
        source= dct["source"], authors= convert_to_list(dct["authors"]) if dct["authors"] is not None else [])
    article.make_chunks(MAX_LEN = len_chunk)
    return article

def solve_window(model_instance, claims : List[str], justifications : List[str]):
    Class ,Confidence = model_instance.verifyClaim(claims, justifications)
    return Class , Confidence


    
def form_batch(claim : str , articles : List[news_article]):
    batch = []
    for article in articles:
        batch.append((claim, article.chunks[0]))

    return batch










def Solver(claim: str, model_instance : BertNLIModel):
    """
    Takes in a claim : str

    returns 
    1) None if the claim does not match any given article in the database
    2) a tuple (answer : str, article : str), where answer is on of "agree", "disagree" etc... and article is the source we base it on 


    """
    if((not isinstance(claim,str)) or (not len(claim) > 0)):
        assert "claim must be a string of non zero length"


    search = {"size": TOP_K_ARTICLES,"query": {"match": {"text": claim}}}
    res = es.search(index= INDEX_NAME,body = json.dumps(search))

    top_hits = res["hits"]["hits"]
    if(len(top_hits) == 0):
        return None



    

    articles = []

    claim_len = len(claim.split())

    for hit in top_hits:
        articles.append(convert_dict_to_article(hit, len_chunk= MAX_LEN - claim_len))
    
    input_batch = form_batch(claim = claim,  articles= articles)
    for b in input_batch:
        print(b[1])
    


    labels , scores = model_instance(input_batch)


    
    return labels, scores , articles

   

    """
    discuss_count = 0 

    for chunk in top_article.chunks:
        Class , Confidence = solve_window(model_instance=  model_instance,claims = [claim],justifications = [chunk])
        print(Class)
        if(Class == "agree" or Class == "disagree"):

            return (Class , chunk)
        elif(Class == "discuss"):
            discuss_count += 1
    
    if(discuss_count > len(top_article.chunks)//2):
        return ("discuss", top_article.text)

    return ("unrelated", "The question is unrelated to any new article we currently have")

    """

    


if __name__ == "__main__":
    print(Solver(" China has not been acting in contradiction to its agreements with India ", model_instance= model_instance))
