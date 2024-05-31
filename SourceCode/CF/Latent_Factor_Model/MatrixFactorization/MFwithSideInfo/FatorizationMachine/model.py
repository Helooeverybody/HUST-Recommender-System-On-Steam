import torch
import torch.nn as nn
class FactorizationMachine(nn.Module):
    def __init__(self,num_input, num_factor, padding_idx):
        super(FactorizationMachine, self).__init__()
        self.embedding = nn.Embedding(num_input + 1, num_factor, padding_idx= padding_idx)
        self.embedding.weight.data.uniform_(-1,1)
        torch.nn.init.xavier_normal_(self.embedding.weight.data,gain = 1e-3)
        self.linear_layer = nn.Embedding(num_input+1, 1, padding_idx= padding_idx)
        self.bias = nn.Parameter(data = torch.rand(1))

    def forward(self, x):
        emb = self.embedding(x)
        pow_of_sum = emb.sum(dim = 1 , keepdim = True).pow(2).sum(dim = 2)
        sum_of_pow = emb.pow(2).sum(dim = 1 , keepdim = True).sum(dim = 2)
        out_inter = 0.5 * (pow_of_sum - sum_of_pow)
        out_lin = self.linear_layer(x).sum(dim = 1)
        out = out_inter + out_lin + self.bias
        return out