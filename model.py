import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import CLIPProcessor, CLIPModel
import lightning as L
from torchvision.transforms import ToPILImage


class CLIPImageClassification(nn.Module):
    def __init__(self, num_classes=101):
        super(CLIPImageClassification, self).__init__()
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.fc = nn.Sequential(
            nn.Linear(num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
       
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images, texts):
        images = [ToPILImage()(image) for image in images]
        inputs = self.processor(texts, images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs).logits_per_image
        outputs = self.fc(outputs)

    def to(self, device):
        self.device = device
        self.fc.to(self.device)
        self.model.to(self.device)
        return self
    
class CLIPWrapper(L.LightningModule):
    def __init__(self, model):
        super(CLIPWrapper, self).__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        images, texts, labels = batch
        outputs = self.model(images, texts)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
    