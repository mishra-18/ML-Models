import torch
from einops.layers.torch import Rearrange
import torch.nn as nn
from vit import PatchEmbedding, Block

def random_masking(x, mask_ratio):
        """
        X: (B T C) 
        random masking to create randomly shuffled unmasked patches
        """

        B, T, D = x.shape  
        len_keep = int(T * (1 - mask_ratio))
        
        # creating noise of shape (B, T) to latter generate random indices
        noise = torch.rand(B, T, device=x.device)  
        
        # sorting the noise, and then ids_shuffle to keep the original indexe format
        ids_shuffle = torch.argsort(noise, dim=1)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # gathering the first few samples
        ids_keep = ids_shuffle[:, :len_keep]
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, T], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x, mask, ids_restore



class MaskedAutoEncoder(nn.Module):
    def __init__(self, emb_size=1024, decoder_emb_size=512, patch_size=16, num_head=16, encoder_num_layers=24, decoder_num_layers=8, in_channels=3, img_size=224):
        super().__init__()      
        self.patch_embed = PatchEmbedding(emb_size = emb_size)
        self.decoder_embed = nn.Linear(emb_size, decoder_emb_size)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2 + 1, decoder_emb_size), requires_grad=False)
        self.decoder_pred = nn.Linear(decoder_emb_size, patch_size**2 * in_channels, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_emb_size))
        self.encoder_transformer = nn.Sequential(*[Block(emb_size, num_head) for _ in range(encoder_num_layers)])
        self.decoder_transformer = nn.Sequential(*[Block(decoder_emb_size, num_head) for _ in range(decoder_num_layers)])
        self.project = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=patch_size**2 * in_channels, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
    def encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
    
        cls_token = x[:, :1, :]
        x = x[:, 1:, :] 
        
        x, mask, restore_id = self.random_masking(x, mask_ratio)
        
        x = torch.cat((cls_token, x), dim=1)
        
        x = self.encoder_transformer(x)
        
        return x, mask, restore_id
        
    def decoder(self, x, restore_id):
        
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], restore_id.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) 
        x_ = torch.gather(x_, dim=1, index=restore_id.unsqueeze(-1).repeat(1, 1, x.shape[2]))  
        x = torch.cat([x[:, :1, :], x_], dim=1)  

        # add pos embed
        x = x + self.decoder_pos_embed

        x = self.decoder_transformer(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        
        return x

    def loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, patch*patch*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.project(imgs)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 
    
        loss = (loss * mask).sum() / mask.sum()  
        return loss

    def forward(self, img):
        mask_ratio = 0.75

        x, mask, restore_ids = self.encoder(img, mask_ratio)
        pred = self.decoder(x, restore_ids) 
        loss  = self.loss(img, pred, mask) 
        return loss, pred, mask
    

if __name__ == '__main__':
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(1, 3, 224, 224).to(device)
    model = MaskedAutoEncoder().to(device)
    print(model(x)[1].shape)
