import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class SGdataloader:
    
    def __init__(self, text, batch_size, window_size):
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
        
        while len(batch_X) < self.batch_size:
            sentence = self.text
            
            word_X = sentence[self.w_pointer]
            start = max(0, self.w_pointer - self.window_size)
            word_y = sentence[start : self.w_pointer] + \
                     sentence[self.w_pointer + 1 : self.w_pointer + self.window_size]
            word_y = pad_sequence(word_y, self.window_size*2)
            batch_X.append(word_X)
            batch_y.append(word_y)
            self.w_pointer += 1
            if self.w_pointer >= self.max_w_pointer:
                raise StopIteration
        
        batch_X = torch.tensor(batch_X, dtype=torch.long, device=device)
        batch_y = torch.tensor(batch_y, dtype=torch.long, device=device)
        return batch_X, batch_y
    
# モデル
class SG(nn.Module):
    
    def __init__(self, vocab_size, embedding_size):
        super(SG, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.l1 = nn.Linear(self.embedding_size, self.vocab_size, bias=False)
        
    def forward(self, batch_X, batch_y):
        emb_X = self.embedding(batch_X)
        lin_X = self.l1(emb_X)
        log_prob_X = F.log_softmax(lin_X, dim=-1)
        log_prob_X = torch.gather(log_prob_X, 1, batch_y)
        log_prob_X = log_prob_X * (batch_y != PAD).float()
        loss = log_prob_X.sum(1).mean().neg()
        return loss
