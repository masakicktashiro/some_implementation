import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class CBOW_data_loader:
    
    def __init__(self, text, window_size, batch_size):
        self.text = text
        self.window_size = window_size
        self.batch_size = batch_size
        self.w_pointer = 0
        self.max_w_pointer = len(text)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch_X = []
        batch_y = []
        while len(batch_X) < self.batch_size:
            batch_y.append(self.text[self.w_pointer])
            start = max(0, self.w_pointer - self.window_size)
            wd_X = self.text[start : self.w_pointer] + \
                    self.text[self.w_pointer + 1 : self.w_pointer + self.window_size + 1]
            wd_X = pad_sequence(wd_X, self.window_size * 2)
            batch_X.append(wd_X)
            self.w_pointer += 1
            if self.w_pointer >= self.max_w_pointer:
                raise StopIteration
                
        batch_X = torch.tensor(batch_X).to(device).long()
        batch_y = torch.tensor(batch_y).to(device).long()
        return batch_X, batch_y
    
class CBOW(nn.Module):
    
    def __init__(self, vocab_size, embedding_size):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.linear = nn.Linear(self.embedding_size, self.vocab_size)
        
    def forward(self, batch_X, batch_y):
        vec_X = self.embedding(batch_X) #(batch_size, window_size * 2, embedding_size)
        vec_X = vec_X * (batch_X != PAD).float().unsqueeze(-1) 
        sum_X = torch.sum(vec_X, dim=1) 
        lin_X = self.linear(sum_X) # (batch_size, vocab_size)
        log_prob_X = F.log_softmax(lin_X, dim=-1) # (batch_size, vocab_size)
        loss = F.nll_loss(log_prob_X, batch_y)
        return loss
