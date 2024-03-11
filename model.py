import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import CLIPProcessor, CLIPModel
import lightning as L
from torchvision.transforms import ToPILImage


class CLIPImageClassification(nn.Module):
    """
    CLIP model for image classification
    there are two approaches to use CLIP for image classification
    1. use image embedding and text embedding to calculate similarity
    2. use image embedding to classify image
    """
    def __init__(self, num_classes=101, similarity=False):
        super(CLIPImageClassification, self).__init__()
        self.similarity = similarity
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        if self.similarity:
            self.fc_for_text = nn.Sequential(
                nn.Linear(512, 512), # make hidden dimention 1024
                nn.GELU(),
            )
        else:
            self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, num_classes),
            )
        self.fc_for_image = nn.Sequential( 
            nn.Linear(512, 512),
            nn.GELU(),
        )
        
       
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images, texts):
        inputs = self.processor(texts, images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        
        if self.similarity:
            text_emb = self.fc_for_text(outputs.text_embeds) 
            image_text_similarity = (image_emb @ text_emb.T) 
            text_image_similarity = image_text_similarity.T
            ground_truth = torch.arange(0, image_emb.shape[0], dtype=torch.long, device=self.device)
            return image_text_similarity, text_image_similarity, ground_truth
        else:
            image_emb = self.fc_for_image(outputs.image_embeds)
            return self.fc(image_emb)

    def to(self, device):
        self.device = device
        if self.similarity:
            self.fc_for_text.to(self.device)
        else:
            self.fc.to(self.device)
        self.fc_for_image.to(self.device)
        self.model.to(self.device)
        return self
    
class CLIPWrapper(L.LightningModule):
    def __init__(self, model):
        super(CLIPWrapper, self).__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        images, texts, labels = batch
        outputs = self.model(images, texts)
        if self.model.similarity:
            image_loss = F.cross_entropy(outputs[0], outputs[2])
            text_loss = F.cross_entropy(outputs[1], outputs[2])
            loss = (image_loss + text_loss) / 2
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
    