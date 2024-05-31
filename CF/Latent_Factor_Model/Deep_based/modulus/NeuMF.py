import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class MLP_layer(nn.Module):
    def __init__(self,embed_dim):
        super(MLP_layer,self).__init__()
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features =embed_dim, out_features = 64)
        self.dense2 = nn.Linear(in_features = 64, out_features = 32)
        self.dense3 = nn.Linear(in_features = 32, out_features = 8)
        
    def forward(self, user_embed, item_embed):
        concat_features = torch.cat((user_embed,item_embed), dim = 1)
        x = self.dense1(concat_features)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        return x

class NeuralMF(nn.Module):
    ''' NeuMF include two parts elementwise and MLP layers'''
    
    def __init__(self,num_user, num_item, latent_mlp, latent_mf):
        super(NeuralMF,self).__init__()
        self.embedding_user_mlp = nn.Embedding(num_embeddings = num_user,embedding_dim = latent_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings = num_item,embedding_dim = latent_mlp)
    
        self.embedding_user_mf = nn.Embedding(num_embeddings = num_user, embedding_dim = latent_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings = num_item, embedding_dim = latent_mf)
        
        self.MLPlayer = MLP_layer(latent_mlp*2)
        self.dense = nn.Linear(in_features = 16, out_features = 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,user_idx,item_idx):
        user_embed_mlp = self.embedding_user_mlp(user_idx)
        item_embed_mlp = self.embedding_item_mlp(item_idx)
        
        user_embed_mf = self.embedding_user_mf(user_idx)
        item_embed_mf = self.embedding_item_mf(item_idx)
        element_wise = user_embed_mf*item_embed_mf
        MLP = self.MLPlayer(user_embed_mlp,item_embed_mlp)
        vector_cocat = torch.cat((element_wise,MLP),dim = 1)

        x = self.dense(vector_cocat)
        x = self.sigmoid(x)
        return x 

def getmodel():
    num_user = 57973
    num_item = 49628
    latent_mlp = 10
    latent_mf = 8
    model = NeuralMF(num_user, num_item, latent_mlp, latent_mf)
    return model 

if __name__ == '__main__':
    pass 
