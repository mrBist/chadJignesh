import numpy as np
from elasticsearch import Elasticsearch
from NEWS.data_indexer import INDEX_NAME,ES_HOST
from NEWS.fake_news_detection import FakeNewsDetector
import json
from spacy.lang.en import English

es = Elasticsearch(hosts= [ES_HOST])
#model_instance = FakeNewsDetector("finetuned_BERT_epoch_5.pt")
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))
MAX_LEN = 512


def solve_window(model_instance, claim : str, justification : str):
    Class ,Confidence = model_instance.verifyClaim(claim, justification)
    return Class , Confidence


def make_windows(text : str):
    doc = nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]
    total_sents = len(sentences)

    # Checking if any sentence has a length > MAX_LENGTH if it has then truncate the sentence 


    truncate = lambda sent : sent[:MAX_LEN]
    sentences = list(map(truncate,sentences))

    
    sent_lengths = [len(s.split()) for s in sentences]
    
    chunks = []

    window_start = 0
    window_end = 0
    word_count = 0
    while window_end < total_sents:

        if(sent_lengths[window_end] + word_count > 512):
            temp_str = ""
            for i in range(window_start, window_end):
                temp_str += sentences[i]
            chunks.append(temp_str)
            
            window_start += 1
            word_count -= sent_lengths[window_start]
        else:
            word_count += sent_lengths[window_end]
            window_end += 1
    
    
    
    temp_str = ""
    for i in range(window_start, window_end):
        temp_str += sentences[i]
        
    chunks.append(temp_str)
    return chunks




def Solver(claim: str, model_instance : FakeNewsDetector):
    """
    Takes in a claim : str

    returns 
    1) None if the claim does not match any given article in the database
    2) a tuple (answer : str, article : str), where answer is on of "agree", "disagree" etc... and article is the source we base it on 


    """
    if((not isinstance(claim,str)) or (not len(claim) > 0)):
        assert "claim must be a string of non zero length"



    search = {"size": 4,"query": {"match": {"text": claim}}}
    res = es.search(index= INDEX_NAME,body = json.dumps(search))

    top = res["hits"]["hits"]

    if(len(top) == 0):
        return None

    top = top[0]
    text  = top["_source"]["text"]
    title = top["_source"]["title"]
    

    chunks = make_windows(text= text)


    discuss_count = 0 

    for chunk in chunks:
        Class , Confidence = solve_window(model_instance = model_instance,claim= claim,justification= chunk)
        print(Class)
        if(Class == "agree" or Class == "disagree"):
            return (Class , chunk)
        elif(Class == "discuss"):
            discuss_count += 1
    
    if(discuss_count > len(chunks)//2):
        return ("discuss", text)

    return ("unrelated", "The question is unrelated to any new article we currently have")

    

    


if __name__ == "__main__":
    print(Solver("No one died in chine due to corona virus"))
