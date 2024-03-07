# importing libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
# reading txt file (encode decode)
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
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
embed_size = 64
dropout = 0
num_head = 4
num_layers = 4
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'False' else val  
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
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

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
            # crop idx to the last block_size tokens
            crop_idx= idx[:, -block_size:]
            # get the predictions
            logits, loss = self(crop_idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(65).to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# training the model, cause I won't give up without a fight 
batch_size = 32
for epoch in range(5000):

    # Printing the Training and Validation Loss
    if epoch%1000==0: 
        m.eval()
        Loss= 0.0
        Val_Loss = 0.0
        for k in range(200):
            x, y = get_batch(True)
            
            val_ , val_loss = m(x, y)
            x1, y1 = get_batch(False)

            _, train_loss = m(x1, y1)            
            Loss += train_loss.item()
            Val_Loss += val_loss.item()
        avg_loss = Val_Loss/(k+1)

        avg_train_loss = Loss/(k+1)
        m.train()
        
        print("Epoch: {} \n The validation loss is:{}    The Loss is:{}".format(epoch, avg_loss, avg_train_loss))
    # Forward
    data, target = get_batch(False)
    logits, loss = m(data, target)
    #Backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# for torch.trill
# block_size = 1 

print("".join(decode(m.generate(torch.zeros([1,1], dtype=torch.long) , max_new_tokens=2000)[0].tolist())))