import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
     def __init__(self, in_ch, embed_size, patch_size):
         super().__init__()
         self.patch_embed = nn.Sequential(
             nn.Conv2d(in_ch, embed_size, kernel_size = patch_size, stride = patch_size),
             Rearrange('b e h w -> b (h w) e')
         )

     def forward(self, x):
         return self.patch_embed(x)
     

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_head = num_heads
        self.qkv = nn.Linear(embed_dim,3*embed_dim)
        self.att_dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.qkv(x)

        x = rearrange(x, 'b n (e k) -> b n e k', k = 3)

        x = rearrange(x, 'b n (e H) k -> b H n e k', H = self.num_head)

        q, k, v = x.chunk(3, dim=-1)
        q, k, v = q.squeeze(-1), k.squeeze(-1), v.squeeze(-1)

        att_score = q@k.transpose(2, 3)/self.num_head**0.5
        wei = F.softmax(att_score, dim=-1)
        att_score = wei@v
        x = rearrange(att_score, 'b H n e -> b n (H e)')
        x = self.att_dropout(x)
        return x
  
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, dropout=0.1):
        super().__init__()
        self.mhsa = MultiheadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, 4*embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

        # Layer Scale
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, embed_dim).fill_(init_eps)
        self.layer_scale = nn.Parameter(scale)

    def forward(self, x):
        x = x + self.layer_scale*self.mhsa(self.ln(x))
        x = x + self.layer_scale*self.ff(self.ln(x))
        return x
  
class Cait(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size, num_heads, depth, cls_depth,  image_size, num_classes):
        super().__init__()
        self.patch_transformer = nn.Sequential(*[Block(embed_dim, num_heads, depth) for _ in range(depth)])
        self.cls_transformer = nn.Sequential(*[Block(embed_dim, num_heads, depth) for _ in range(cls_depth)])

        self.patch_embed = PatchEmbedding(in_ch, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.rand((image_size//patch_size)**2, embed_dim))

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.patch_transformer(x)

        cls_token = repeat(self.cls_token, ' () n e  -> b n e', b=b)

        x = torch.cat([cls_token, x], dim=1)

        x = self.cls_transformer(x)

        x = x[:, 1, :]

        x = self.mlp(x)

        return x
    
if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(1, 3, 224, 224).to(device)
    model = Cait(in_ch=3,
            embed_dim=256,
            patch_size=16,
            num_heads=4,
            depth=24,
            cls_depth=2,
            image_size=224,
            num_classes=10).to(device)
    
    print(model(x).shape)