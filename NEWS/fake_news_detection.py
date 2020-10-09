import torch
import torch.nn as nn
import re
import numpy as np
from transformers import BertTokenizer,BertModel



class FakeNewsDetector():
    
    def __init__(self, model_path):
        self.model_path = model_path
        # self.model = torch.load(model_path, map_location='cpu')
        self.model = BERTModel()
        print("loading model")
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print("model loaded")
        self.tokenizer = BertTokenizer.from_pretrained('sentence-transformers/bert-base-nli-cls-token'
                                            ) 
        self.labels = {0:'disagree', 1:'agree', 2:'discuss', 3:'unrelated'}
        self.num_classes = len(self.labels)
        self.max_len = 512
        print("init done")
    
    def verifyClaim(self, claim, reference):
        
        #print("claim: ", claim)
        #print("reference: ", reference)
        
        #encode batch of sentences
        encoded_data = self.tokenizer.encode_plus(
            reference,
            claim,
            add_special_tokens=True,   
            max_length=self.max_len,
            truncation = True
        )   
        
        # get ids and attention masks
        ids = encoded_data['input_ids']
        token_type_ids = encoded_data['token_type_ids']
        mask = encoded_data['attention_mask']
        
        #pad to max length
        padding_len = self.max_len - len(ids)
        ids = ids + ([0]*padding_len)
        token_type_ids = token_type_ids + ([0]*padding_len)
        mask = mask + ([0]*padding_len)
       
        # convert to tensor
        ids = torch.LongTensor(ids).unsqueeze(0)
        mask = torch.LongTensor(mask).unsqueeze(0)
        token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0)
        
        # pass through the model
        outputs = self.model(ids, mask, token_type_ids)
        
        # get probability using softmax
        outputs = torch.softmax(outputs, dim=1).cpu().detach().numpy()[0]  # shape(6,)
        predicted_classes = np.argmax(outputs)
        confidence_scores = np.max(outputs)
        
        classes = self.labels[predicted_classes]
        
        return classes, confidence_scores


class BERTModel(nn.Module):
  
  def __init__(self):
    super(BERTModel,self).__init__()
    self.num_classes = 4
    self.bert = BertModel.from_pretrained('sentence-transformers/bert-base-nli-cls-token')
    self.dropout = nn.Dropout(0.2)
    self.out = nn.Linear(768,self.num_classes)
  
  def forward(self,ids,mask,token_type_ids):
    _, o2 = self.bert(ids, attention_mask=mask,token_type_ids=token_type_ids)
    bo = self.dropout(o2)
    return self.out(bo)
