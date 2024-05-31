import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class NCF(nn.Module):
    ''' Neural Collaborative filtering (NCF) architecture based on MLP and embedding matric of user and item 
    learning latent factor of user and item in Black Box'''
    def __init__(self,embed_dim,num_user,num_item):

        super(NCF,self).__init__()
        self.relu = nn.ReLU()
        self.embedding_user = nn.Embedding(num_embeddings = num_user,embedding_dim = embed_dim)
        self.embedding_item = nn.Embedding(num_embeddings = num_item,embedding_dim = embed_dim)
        self.dense1 = nn.Linear(in_features =embed_dim*2, out_features = 64)
        self.dense2 = nn.Linear(in_features = 64, out_features = 32)
        self.dense3 = nn.Linear(in_features = 32, out_features = 16)
        self.dense4 = nn.Linear(in_features = 16, out_features = 1)
        
        self.softmax = nn.Softmax(dim=1) 
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, user, item):
        user_embed = self.embedding_user(user)
        item_embed = self.embedding_item(item)
        concat_features = torch.cat((user_embed,item_embed), dim = 1)
        x = self.dense1(concat_features)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dense4(x)
        x = self.sigmoid(x)
        return x

def getmodel():
    num_user = 57973  # number of user in data
    num_item = 49628 # number of item in data
    num_embed = 8 # number latent factor in embedding matrix

    model = NCF(num_embed,num_user, num_item)    
    return model

if __name__ == '__main__':
    pass 
