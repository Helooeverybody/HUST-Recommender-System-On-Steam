import torch
import torch.nn as nn
import numpy as np
class MFbpr(nn.Module):
    def __init__(self,  num_user, num_item, factors = 30, reg =0.1, init_mean = 1, init_stdev = 0.5):
        super(MFbpr, self).__init__()
        self.factors = factors
        self.reg = reg
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.U = torch.normal(mean = init_mean * torch.ones(num_user, factors), std = init_stdev).requires_grad_()
        self.V = torch.normal(mean = init_mean * torch.ones(num_item, factors), std = init_stdev).requires_grad_()
    def forward(self, u, i, j):
        y_ui = torch.diag(torch.mm(self.U[u], self.V[i].t()))
        y_uj = torch.diag(torch.mm(self.U[u], self.V[j].t()))
        regularizer = self.reg * (torch.sum(self.U[u] ** 2) + torch.sum(self.V[i] ** 2) + torch.sum(self.V[j] ** 2))
        loss = regularizer - torch.sum(torch.log2(torch.sigmoid(y_ui - y_uj)))
        return y_ui, y_uj, loss
    def predict(self, u, i):
        return np.inner(self.U[u].detach().numpy(), self.V[i].detach().numpy())
