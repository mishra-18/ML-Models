"""
The following code is updated version of bigram model: https://github.com/mishra-18/ML-Models/blob/main/Transformer/bigram.py
Implements Key-Value Cache to the attention mechanism and is only responsible for performing inference.
"""
# importing libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
# reading txt file (encode decode) /* downloaded from: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt */
text = open('input.txt', 'r',).read()
vocab = sorted(list(set(text)))
encode = lambda s: [vocab.index(c) for c in s]
decode = lambda l: [vocab[c] for c in l]
# splitting the train and val dataset
x = int(0.9*len(text))
text = torch.tensor(encode(text), dtype=torch.long)
train, val = text[:x], text[x:]
# creating a get_batch function to randomly load data from text in shape (batch_size, ,vocab_size(8))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

batch_size = 4 
# NOTE: Keep the block_size and embed_size same as the model weights being loaded
block_size = 1024 # maximum context length for predictions
embed_size = 256  
dropout = 0
num_head = 4
num_layers = 4

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'train' else val  
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
xb, yb = get_batch('train')
# Attention Head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.k_cache = None
        self.v_cache = None
        self.cache_index = 0

    def forward(self, x):
        B, T, C = x.shape # B, 1, C
        # print("TOken", T)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Initialize cache if empty
        if self.k_cache is None or self.v_cache is None:
            # Initialize cache with fixed size
            self.k_cache = torch.zeros(B, block_size, self.head_size, device=x.device)
            self.v_cache = torch.zeros(B, block_size, self.head_size, device=x.device)
            self.cache_index = 0
        # start = time.time()

        # Update cache in-place
        if self.cache_index + T <= block_size:
            self.k_cache[:, self.cache_index:self.cache_index + T, :] = k
            self.v_cache[:, self.cache_index:self.cache_index + T, :] = v
        else:
            shift = self.cache_index + T - block_size
            self.k_cache[:, :-shift, :] = self.k_cache[:, shift:, :].clone()
            self.v_cache[:, :-shift, :] = self.v_cache[:, shift:, :].clone()
            self.k_cache[:, -T:, :] = k
            self.v_cache[:, -T:, :] = v

        # Update cache index
        self.cache_index = min(self.cache_index + T, block_size)

        # Attention
        wei = q@self.k_cache.transpose(2, 1)/self.head_size**0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=2)    # (B, block_size, block_size)
        wei = self.dropout(wei)
        out = wei@self.v_cache

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_head):
        super().__init__()
        self.sa_head = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.dropout = nn.Dropout(dropout)
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
              nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)
    
class Block(nn.Module):
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
    
    # super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.possitional_embedding = nn.Embedding(block_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.block = nn.Sequential(*[Block(embed_size, num_head) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        ps = self.possitional_embedding(torch.arange(T, device=device))
        x = logits + ps    #(B, T, C)
        logits = self.block(x)     #(B, T, c)
        logits = self.linear(self.layer_norm(logits)) # This suppose to map between head_size and Vocab_size 
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx) # logits shape: B, 1, C
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1).to(device) # (B, 1)
            # No need to concat as we are only passing one token at a time
            print("".join(decode([idx_next.item()])), end='')

            idx = idx_next

        return idx

m_kv_cache = BigramLanguageModel(65).to(device)
# NOTE: Assumes you trained the bigram model with the same block_size, embed_size as above and saved the weights as bigram_best.pth 
m_kv_cache.load_state_dict(torch.load("bigram_best.pth"))

if __name__ == '__main__':
    print("".join(decode(m_kv_cache.generate(torch.zeros([1,1], dtype=torch.long).to(device) , max_new_tokens=100)[0].tolist())))