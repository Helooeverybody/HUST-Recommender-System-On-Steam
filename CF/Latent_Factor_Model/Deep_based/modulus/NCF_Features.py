import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class NCF_Feature(nn.Module):
    def __init__(self,num_tag,dim_user,dim_item):
        super(NCF_Feature,self).__init__()
        self.linear_user = nn.Linear(in_features = dim_user, out_features = 10)
        self.linear_item = nn.Linear(in_features = dim_item, out_features = 10)
        
        self.embedding_user_tag = nn.Embedding(num_embeddings=num_tag,embedding_dim = 5)
        self.embedding_item_tag = nn.Embedding(num_embeddings = num_tag,embedding_dim = 5)
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(10)

        self.dense1 = nn.Linear(in_features = 30, out_features = 64)
        self.dense2 = nn.Linear(in_features = 64, out_features = 32)
        self.dense3 = nn.Linear(in_features = 32, out_features = 1)
        self.relu = nn.ReLU()

    def forward(self,user_idx,item_idx,user_feature, item_feature,user_tag,item_tag):
        user = self.linear_user(user_feature)
        item = self.linear_item(item_feature)

        user = self.bn1(user)
        item = self.bn2(item)

        # embedding tag 
        embedd_user_tag = torch.zeros(len(user_tag),5)
        for k in range(len(user_tag)):
            nonzero_indices = (user_tag[k] != 0).nonzero(as_tuple=True)[0]
            embeddings = self.embedding_item_tag(nonzero_indices)
            weighted_embeddings = embeddings * (user_tag[k][nonzero_indices].unsqueeze(1)/10)
            embedd_user_tag[k] = torch.sum(weighted_embeddings,dim = 0)

        embedd_item_tag = torch.zeros(len(item_tag),5)
        for k in range(len(item_tag)): 
            one_hot_index = (item_tag[k] == 1).nonzero(as_tuple=True)[0]
            embedd_item_tag[k] = self.embedding_user_tag(one_hot_index)
        new_user = torch.concat((user,embedd_user_tag),dim = 1) 
        new_item = torch.concat((item,embedd_item_tag), dim = 1)   
        
        feature_concat = torch.concat((new_user,new_item),dim = 1)
        x = self.dense1(feature_concat)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)

        x = self.dense3(x)
        x = self.relu(x)
        return x 

def get_model():
    model = NCF_feature(num_tag = 10 ,dim_user= 3,dim_item = 4)

    return model 

if __name__== '__main__':
    pass 
