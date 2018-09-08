import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class GLOVE(nn.Module):
    
    def __init__(self, vocab_size, embedding_size, co_matrix):
        super(GLOVE, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.in_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.out_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.in_bias_ls = autograd.Variable(torch.randn(self.vocab_size).float()) ##初期化
        self.out_bias_ls = autograd.Variable(torch.randn(self.vocab_size).float())
        self.co_matrix = co_matrix
        
    def forward(self, wd_x, wd_y):
        vec_x = self.in_embedding(wd_x) #(batch_size, embedding_size)
        vec_y = self.out_embedding(wd_y) * (wd_y != PAD).float().unsqueeze(-1)  # (batch_size, window_size * 2, embedding_size)
        bias_x = self.in_bias_ls[wd_x].unsqueeze(1).to(device) #(batch_size, 1)
        bias_y = self.out_bias_ls[wd_y].to(device)#(batch_size, window_size)
        target_matrix = torch.tensor([self.co_matrix[wd_x][i, wd_y[i]] for i in range(len(wd_x))]).float().to(device)
        loss = self.f(target_matrix) * (torch.bmm(vec_y, vec_x.unsqueeze(2)).squeeze() + bias_x + bias_y - \
                                                        target_matrix) ** 2 #(batch_size)
        
        loss = (loss * (wd_y != PAD).float()).sum(dim=-1)
        loss = loss.sum(dim=-1)
        return loss
        
    def f(self, x, x_max=100, alpha=0.75):
        x = torch.tensor(x).float()
        return torch.where(x < x_max, torch.pow(x / x_max, alpha), torch.ones_like(x)).to(device)

class Glove_data_loader:
    
    def __init__(self, text, batch_size, window_size=5):
        self.text = text
        self.batch_size = batch_size
        self.window_size = window_size
        self.w_pointer = 0
        self.max_w_pointer = len(text)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        batch_X = []
        batch_y = []
        
        while len(batch_X) <= self.batch_size:
            batch_X.append(self.text[self.w_pointer])
            start = max(0, self.w_pointer - self.window_size)
            wd_y = self.text[start : self.w_pointer] + \
                          self.text[self.w_pointer + 1 : self.w_pointer + self.window_size + 1]
            wd_y = pad_sequence(wd_y, self.window_size * 2)
            batch_y.append(wd_y)
            self.w_pointer += 1
            if self.w_pointer >= self.max_w_pointer:
                raise StopIteration
            
        batch_X = torch.tensor(batch_X).long().to(device)
        batch_y = torch.tensor(batch_y).long().to(device)
        return batch_X, batch_y
                
        
def make_co_matrix(text, len_vocab, window_size=5):
    """
    text : list
        idのリスト
    window_size : int
        window_size
    co_matrix : ndarray
        共起行列
    """
    assert isinstance(text, list)
    assert isinstance(text[0], int)
    co_matrix = sparse.lil_matrix((len_vocab, len_vocab))
    for target_ind in range(len_vocab):
        start = max(0, target_ind - window_size)
        target = text[target_ind]
        context = text[start : target_ind] + text[min(target_ind + 1, len_vocab) : min(target_ind + 1 + window_size, len_vocab)]
        for context_ind in context:
            co_matrix[target_ind, context_ind] += 1
    return co_matrix

