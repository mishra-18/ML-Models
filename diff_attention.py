import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # Learnable scaling parameter
        self.eps = eps  # Small constant for numerical stability

    def forward(self, x):
        # Compute root mean square (RMS) of the input
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight

class DiffAttn(nn.Module):
    def __init__(self, num_heads, embed_dim, depth):
        super().__init__()
        self.head_dim = int(embed_dim/num_heads)
        
        self.q_linear = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, self.head_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32))
        
        # mean = 0 (default); std = 0.1
        nn.init.normal_(self.lambda_q1, std=0.1)
        nn.init.normal_(self.lambda_q2, std=0.1)
        nn.init.normal_(self.lambda_k1, std=0.1)
        nn.init.normal_(self.lambda_k2, std=0.1)

        try:
            from apex.normalization import FusedRMSNorm
            self.ln = FusedRMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        except ImportError:
            self.ln = RMSNorm(self.head_dim, eps=1e-5)

    def forward(self, x):
        b, t, d = x.shape # T : Token/Sequence Length

        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Split q and k into two parts
        q1, q2 = torch.chunk(q, 2, dim=-1)
        k1, k2 = torch.chunk(k, 2, dim=-1)
        
        # Compute Attention Scores
        attn1 = q1 @ k1.transpose(-2, -1) / math.sqrt(self.head_dim / 2)
        attn2 = q2 @ k2.transpose(-2, -1) / math.sqrt(self.head_dim / 2)
        
        # Creating a mask as Diff Attn paper trains a decoder only model
        attn_mask = torch.triu(torch.zeros([t, t]).fill_(float("-inf")), diagonal=1)
       
        # Compute Saperate scores
        a1 = f.softmax(attn1+attn_mask / math.sqrt(self.head_dim / 2), dim=-1)
        a2 = f.softmax(attn2+attn_mask / math.sqrt(self.head_dim / 2), dim=-1)
        
        # Compute lmbda dynamically
        self.lmbda = torch.exp(torch.sum(self.lambda_q1*self.lambda_k1, dim=-1)) \
                    -  torch.exp(torch.sum(self.lambda_q2*self.lambda_k2, dim=-1)) + self.lambda_init
        
        diffattn = (a1 - self.lmbda*a2)@v
        attn = (1 - self.lambda_init)*self.ln(diffattn) 

        return attn

class MultiHead(nn.Module):
    def __init__(self, num_heads, embed_dim, depth):
        super().__init__()
        self.attn_heads = nn.ModuleList([DiffAttn(num_heads, embed_dim, depth) for _ in range(num_heads)])
        self.o_linear = nn.Linear(embed_dim, embed_dim, bias=False) 
    def forward(self, x):
        x = torch.cat([attn_head(x) for attn_head in self.attn_heads], dim=-1)
        out = x*self.o_linear(x)
        return out