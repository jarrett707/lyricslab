#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import requests
import tiktoken
import numpy as np
import pandas as pd


# In[16]:


df = pd.read_csv("spotify_millsongdata.csv")
data = df['text'].str.cat(sep='\n')


# In[17]:


n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]


# In[18]:


enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

