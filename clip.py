import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from PIL import Image
from vit import VissionTransformer
import numpy as np
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, texts, image_paths):

        self.image_paths = image_paths
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt") 
        self.transform = torchvision.transforms.ToTensor()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        image = self.transform(image)

        caption, mask = self.inputs[idx].items()
        
        return {
            "image": image,
            "input_ids": caption["input_ids"],
            "mask": mask["attention_mask"]
        }

class TextEncoder(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()
        self.model = DistilBertModel(config=DistilBertConfig())
        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        x = x[:, 0, :] # B, T[cls], E
        
        x = self.projection(x)

        return self.layer_norm(x)
    
class ImageEncoder(nn.Module):
    def __init__(self, base_model, embed_dim, proj_dim):
        super().__init__()

        self.model = base_model

        for param in self.model.parameters():
            param.requires_grad = True

        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        x = self.projection(self.model(x))
        return self.layer_norm(x)
    
class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ViT = VissionTransformer( 
            num_layers=8,
            img_size=224,
            emb_size=768,
            patch_size=16,
            num_head=6,
            num_class=False).to(self.device)
        self.image_encoder = ImageEncoder(base_model=ViT, embed_dim=768, proj_dim=256)
        self.text_encoder = TextEncoder(embed_dim=768, proj_dim=256)

        self.temperature = nn.Parameter(torch.ones([])*np.log(1/7)).to(self.device)

    def forward(self, x):
        I_t = self.image_encoder(x["image"])
        T_t = self.text_encoder(x["input_ids"], x["mask"])

        logits = I_t@T_t.T * torch.exp(self.temperature)

        labels = torch.arange(I_t.size(0)).to(self.device)

        loss_I = F.cross_entropy(logits.T, labels)
        loss_T = F.cross_entropy(logits, labels)

        loss = (loss_I + loss_T)/2.0 

        return loss, logits

if __name__ == '__main__':

    texts = ["This is a sample sentence.", "This is another example."]

    # train_data = CustomDataset(texts, image_path)
    # train_loader = DataLoader(train_data, batch_size, shuffle=True)
    
    # Example Usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    inputs= tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    test = {
    "image" : torch.rand(2, 3, 224, 224).to(device),
    "input_ids" : inputs["input_ids"],
    "mask" : inputs["attention_mask"]
    }

    model = CLIPModel().to(device)
    loss, logits = model(test)
    print("Loss:", loss, "Logits:", logits)