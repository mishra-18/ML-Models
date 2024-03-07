# importing libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
# Attention Head
class Head(nn.Module):
    def __init__(self, head_size, embed_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q@k.transpose(2, 1)/self.head_size**0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=2)    # (B , block_size, block_size)
        wei = self.dropout(wei)
        out = wei@v
        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_head, embed_size):
        super().__init__()
        self.sa_head = nn.ModuleList([Head(head_size, embed_size) for _ in range(num_head)])
        self.dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.sa_head], dim= -1)
        x = self.dropout(self.proj(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
    
        self.ff = nn.Sequential(
              nn.Linear(embed_size, 4*embed_size),
              nn.ReLU(),
              nn.Linear(4*embed_size, embed_size),
              nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.ff(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, embed_size, num_head):
        super().__init__()
        head_size = embed_size // num_head
        self.multihead = MultiHeadAttention(head_size, num_head)
        self.ff = FeedForward(embed_size)
        self.ll1 = nn.LayerNorm(embed_size)
        self.ll2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.multihead(self.ll1(x))
        x = x + self.ff(self.ll2(x))
        return x

if __name__ == '__main__':
    embed_size = 64
    batch_size = 1
    block_size = 8
    num_heads = 4
    x = torch.rand(batch_size, block_size, embed_size) 
    x = AttentionBlock(embed_size=embed_size, num_head=num_heads)(x)
    print(x.shape)