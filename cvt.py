import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class CVTEmbedding(nn.Module):
  def __init__(self, in_ch, embed_dim, patch_size,stride):
    super().__init__()
    self.embed = nn.Sequential(
        nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride)
    )
    self.norm = nn.LayerNorm(embed_dim)
  def forward(self, x):
    x = self.embed(x)
    x = rearrange(x, 'b c h w -> b (h w) c')
    x = self.norm(x)
    return x
  

class MultiHeadAttention(nn.Module):
  def __init__(self, in_dim, num_heads, kernel_size=3, with_cls_token=False):
    super().__init__()
    padding = (kernel_size - 1)//2
    self.forward_conv = self.forward_conv
    self.num_heads = num_heads
    self.with_cls_token = with_cls_token
    self.conv = nn.Sequential(
        nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, padding=padding, stride=1),
        nn.BatchNorm2d(in_dim),
        Rearrange('b c h w -> b (h w) c')
    )
    self.att_drop = nn.Dropout(0.1)

  def forward_conv(self, x):
    B, hw, C = x.shape
    if self.with_cls_token:
      cls_token, x = torch.split(x, [1, hw-1], 1)
    H = W = int(x.shape[1]**0.5)
    x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
    q = self.conv(x)
    k = self.conv(x)
    v = self.conv(x)

    if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

    return q, k, v

  def forward(self, x):
    # x -> B, (H, W), C

    q, k, v = self.forward_conv(x)

    q = rearrange(x, 'b t (d H) -> b H t d', H=self.num_heads)
    k = rearrange(x, 'b t (d H) -> b H t d', H=self.num_heads)
    v = rearrange(x, 'b t (d H) -> b H t d', H=self.num_heads)

    att_score = q@k.transpose(2, 3)/self.num_heads**0.5
    att_score = F.softmax(att_score, dim=-1)
    att_score = self.att_drop(att_score)

    x = att_score@v
    x = rearrange(x, 'b H t d -> b t (H d)')

    return x


class MLP(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.ff = nn.Sequential(
       nn.Linear(dim, 4*dim),
       nn.GELU(),
       nn.Dropout(0.1),
       nn.Linear(4*dim, dim),
       nn.Dropout(0.1)
    )

  def forward(self, x):
     return self.ff(x)


class Block(nn.Module):
  def __init__(self, embed_dim, num_heads, with_cls_token):
    super().__init__()

    self.mhsa = MultiHeadAttention(embed_dim, num_heads, with_cls_token=with_cls_token)
    self.ff = MLP(embed_dim)
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    x = x + self.dropout(self.mhsa(self.norm1(x)))
    x = x + self.dropout(self.ff(self.norm2(x)))
    return x
  

class VissionTransformer(nn.Module):
  def __init__(self, depth, embed_dim, num_heads, patch_size, stride, in_ch=3, cls_token=False):
    super().__init__()

    self.stride = stride
    self.cls_token = cls_token
    self.layers = nn.Sequential(*[Block(embed_dim, num_heads, cls_token) for _ in range(depth)])
    self.embedding = CVTEmbedding(in_ch, embed_dim, patch_size, stride)

    if self.cls_token:
       self.cls_token_embed = nn.Parameter(torch.randn(1, 1, 384))

  def forward(self, x, ch_out=False):
    B, C, H, W = x.shape
    x = self.embedding(x)
    if self.cls_token:
      cls_token = repeat(self.cls_token_embed, ' () s e -> b s e', b=B)

      x = torch.cat([cls_token, x], dim=1)

    x = self.layers(x)

    if not ch_out:
       x = rearrange(x, 'b (h w) c -> b c h w', h=(H -1)//self.stride, w=(W-1)//self.stride)
    return x
  

class CvT(nn.Module):
  def __init__(self, embed_dim, num_class):
    super().__init__()

    self.stage1 = VissionTransformer(depth=1,
                                     embed_dim=64,
                                     num_heads=1,
                                     patch_size=7,
                                     stride=4,
                                     )
    self.stage2 = VissionTransformer(depth=2,
                                     embed_dim=192,
                                     num_heads=3,
                                     patch_size=3,
                                     stride=2,
                                     in_ch = 64)
    self.stage3 = VissionTransformer(depth=10,
                                     embed_dim=384,
                                     num_heads=6,
                                     patch_size=3,
                                     stride=2,
                                     in_ch=192,
                                     cls_token=True)
    self.ff = nn.Sequential(
        nn.Linear(6*embed_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, num_class)
    )
  def forward(self, x):
    x = self.stage1(x)
    x = self.stage2(x)
    x = self.stage3(x, ch_out=True)
    x = x[:, 1, :]
    x = self.ff(x)
    return x  
    


if __name__ == '__main__':
  # Usage example
  device = 'cuda' if torch.cuda.is_available() else 'cpu'  
  x = torch.randn(1, 3, 224, 224).to(device) 
  embed_size = 64
  num_class = 10
  model = CvT(embed_size, num_class).to(device)
  print(model(x).shape)