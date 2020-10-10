import os


from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))


INDEX_NAME = "newsdata"
DOC_TYPE = "articles"
ES_HOST = {"host" : "localhost", "port" : 9200}
MAX_LEN = 512


TOP_K_ARTICLES = 4




class news_article:
    def __init__(self,title : str, text : str, authors : list, source : str):
        self.title = title
        self.text = text
        self.authors = authors
        self.source = source
    def __str__(self) -> str:
        return self.title

    def make_chunks(self, MAX_LEN = 512):
        doc = nlp(self.text)
        sentences = [sent.string.strip() for sent in doc.sents]
        total_sents = len(sentences)

        # Checking if any sentence has a length > MAX_LENGTH if it has then truncate the sentence 


        truncate = lambda sent : sent[:MAX_LEN]
        sentences = list(map(truncate,sentences))

        
        sent_lengths = [len(s.split()) for s in sentences]
        
        self.chunks = []

        window_start = 0
        window_end = 0
        word_count = 0
        while window_end < total_sents:

            if(sent_lengths[window_end] + word_count > MAX_LEN):
                temp_str = ""
                for i in range(window_start, window_end):
                    temp_str += sentences[i]
                self.chunks.append(temp_str)
                
                window_start += 1
                word_count -= sent_lengths[window_start]
            else:
                word_count += sent_lengths[window_end]
                window_end += 1
        
        
        
        temp_str = ""
        for i in range(window_start, window_end):
            temp_str += sentences[i]
            
        self.chunks.append(temp_str)


