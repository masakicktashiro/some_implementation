import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

def pad_sequence(seq, max_len):
    seq += [PAD for i in range(max_len - len(seq))]
    return seq

class SGNS_data_loader:
    
    def __init__(self, text, batch_size, window_size, n_negative, weights = None):
        self.text = text
        self.batch_size = batch_size
        self.window_size = window_size
        self.w_pointer = 0
        self.w_max_pointer = len(text)
        self.n_negative = n_negative
        self.weights = torch.FloatTensor(weights)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        batch_X = []
        batch_y = []
        batch_N = []
        
        while len(batch_X) < self.batch_size:
            batch_X.append(self.text[self.w_pointer])
            start = max(0, self.w_pointer - self.window_size)
            word_y = self.text[start : self.w_pointer] + \
                           self.text[self.w_pointer + 1 : self.w_pointer + self.window_size]
            word_y = pad_sequence(word_y, self.window_size * 2)
            batch_y.append(word_y)
            batch_N.append(torch.multinomial(self.weights, self.n_negative).tolist())
            self.w_pointer += 1
            if self.w_max_pointer <= self.w_pointer:
                raise StopIteration
        batch_X = torch.tensor(batch_X).long().to(device) #(batch_size, )
        batch_y = torch.tensor(batch_y).long().to(device) #(batch_size, window_size*2)
        batch_N = torch.tensor(batch_N).long().to(device) #(batch_size, n_negative)
        return batch_X, batch_y, batch_N
        
class SGNS(nn.Module):
    
    def __init__(self, vocab_size, embedding_size):
        super(SGNS, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.in_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.out_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        
    def forward(self, batch_X, batch_y, batch_N):
        embed_X = self.in_embedding(batch_X).unsqueeze(-1) #(batch_size, embedding, 1)
        embed_y = self.out_embedding(batch_y) #(batch_size, window_size*2, embedding)
        embed_N = self.out_embedding(batch_N) #(batch_size, n_negative, embedding)
        x_y = torch.bmm(embed_y, embed_X) #(batch_size, window_size*2, 1)
        x_n = torch.bmm(embed_N, embed_X) #(batch_size, n_negative, 1)
        x_y_loss = x_y.squeeze().sigmoid().float().log()
        x_n_loss = x_n.squeeze().neg().sigmoid().float().log()
        x_y_loss = (x_y_loss * (batch_y != PAD).float()).sum()
        return - (x_n_loss + x_y_loss).mean()
