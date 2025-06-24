import torch
import numpy as np
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
from backend.config import settings
import os

class ImageEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(self.device)
    
    def embed(self, image_path: str) -> np.ndarray:
        """Mengubah gambar menjadi vektor"""
        img = Image.open(image_path)
        inputs = self.feature_extractor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Gunakan [CLS] token sebagai representasi gambar
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()