import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import CLIPProcessor, CLIPModel
import lightning as L
from torchvision.transforms import ToPILImage


class CLIPImageClassification(nn.Module):
    def __init__(self, num_classes=101, batch_size=32):
        super(CLIPImageClassification, self).__init__()
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.fc_for_text = nn.Sequential(
            nn.Linear(512, 1024), # make hidden dimention 1024
            nn.GELU(),
        )
        self.fc_for_image = nn.Sequential( 
            nn.Linear(512, 1024),
            nn.GELU(),
        )
       
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images, texts):
        inputs = self.processor(texts, images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        text_emb = self.fc_for_text(outputs.text_embeds) 
        image_emb = self.fc_for_image(outputs.image_embeds)
        image_text_similarity = (image_emb @ text_emb.T) 
        text_image_similarity = image_text_similarity.T
        ground_truth = torch.arange(0, image_emb.shape[0], dtype=torch.long, device=self.device)
        return image_text_similarity, text_image_similarity, ground_truth

    def to(self, device):
        self.device = device
        self.fc_for_image.to(self.device)
        self.fc_for_text.to(self.device)
        self.model.to(self.device)
        return self
    
class CLIPWrapper(L.LightningModule):
    def __init__(self, model):
        super(CLIPWrapper, self).__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        images, texts, labels = batch
        outputs = self.model(images, texts)
        images_text_similarity, text_images_similarity, ground_truth = outputs
        image_loss = F.cross_entropy(images_text_similarity, ground_truth)
        text_loss = F.cross_entropy(text_images_similarity, ground_truth)
        loss = (image_loss + text_loss) / 2
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
    