#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import torch
#import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm


# In[ ]:





# In[5]:


import json
reviews=[]
reviewerID=[]
asin = []
rating = []
with open('./reviews_Electronics_5.json') as f:
  for line in tqdm(f):
    dict_rev = json.loads(line.strip())
    reviews.append(dict_rev['reviewText'])
    reviewerID.append(dict_rev['reviewerID'])
    asin.append(dict_rev['asin'])
    rating.append(dict_rev['overall'])


# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}


# In[6]:


data_df=pd.DataFrame(zip(reviewerID, asin, reviews, rating), columns=['ReviewerID','ASIN','Reviews','Ratings'])
data_df.insert(4,'UserID',value=[0]*len(reviews))
data_df.insert(5,'ItemID',value=[0]*len(reviews))


# In[ ]:





# In[ ]:





# In[7]:


usr_buffer={}
itm_buffer={}
for r in tqdm(range(len(data_df))):
  u = data_df['ReviewerID'][r]
  i = data_df['ASIN'][r]
  if u in usr_buffer:
    data_df['UserID'][r] = usr_buffer[u]
  else:
    usr_buffer[u]=len(usr_buffer)
    data_df['UserID'][r] = usr_buffer[u]
  if i in itm_buffer:
    data_df['ItemID'][r] = itm_buffer[i]
  else:
    itm_buffer[i]=len(itm_buffer)
    data_df['ItemID'][r] = itm_buffer[i]



# In[8]:


data_df.to_csv('data_df.csv', index=False)


# In[ ]:


data_df['UserID'].to_csv('userid.txt', index=False, header=False)
data_df['ItemID'].to_csv('itemid.txt', index=False, header=False)


# In[19]:


print(len(usr_buffer))
print(len(itm_buffer))
import nltk
from nltk.corpus import wordnet as wn

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


# In[39]:


reviews_file = open('review.txt','w')
valid_concepts = open('concepts.txt','w')


# In[40]:


from nltk.tokenize import word_tokenize
vocab = {}
#concepts = []
#reviews = []
iterations=0
for r in tqdm(data_df['Reviews']):
  tokens = word_tokenize(r)
  tokens = [word for word in tokens if word.isalpha()]
  pos_tagged = nltk.pos_tag(tokens)
  str_review=''
  str_concept=''
  for i,t in enumerate(tokens):
    str_review+=str(t)
    str_review+='\t'
    if pos_tagged[i][1]=='NN':
      str_concept+=str(t)
      str_concept+='\t'
    if t in vocab:
      vocab[t] += 1
    else:
      vocab[t] = 1
  #concepts.append(str_concept)
  valid_concepts.write(str_concept+'\n')
  #reviews.append(str_review)
  reviews_file.write(str_review+'\n')

valid_concepts.close()
reviews_file.close()



# In[ ]:





# In[ ]:


vocab=dict(sorted(vocab.items(), key=lambda x: x[1],reverse=True))
with open('./vocab.txt','w') as f:
  for x in vocab:
    f.write(str(x)+'\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
