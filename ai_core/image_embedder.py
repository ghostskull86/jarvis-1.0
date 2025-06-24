import torch
import numpy as np
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
from backend.config import settings
from typing import List, Optional
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageEmbedder:
    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k"):
        """
        Inisialisasi Image Embedder dengan model Vision Transformer (ViT)
        
        Args:
            model_name: Nama model pretrained dari HuggingFace
                       Default: google/vit-base-patch16-224-in21k
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Menggunakan device: {self.device} untuk image embedding")
        
        try:
            # Load model dan feature extractor
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            self.model = ViTModel.from_pretrained(model_name).to(self.device)
            logger.info(f"Berhasil memuat model {model_name}")
        except Exception as e:
            logger.error(f"Gagal memuat model: {str(e)}")
            raise

    def embed(self, image_path: str) -> Optional[np.ndarray]:
        """
        Mengubah gambar menjadi vektor embedding
        
        Args:
            image_path: Path ke file gambar
        
        Returns:
            numpy array dengan shape (embedding_dim,) atau None jika gagal
        """
        try:
            # Buka gambar dan konversi ke RGB
            img = Image.open(image_path).convert('RGB')
            
            # Ekstrak fitur
            inputs = self.feature_extractor(images=img, return_tensors="pt").to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Gunakan [CLS] token sebagai representasi gambar
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            logger.debug(f"Berhasil generate embedding untuk {image_path}")
            return embedding
            
        except Exception as e:
            logger.error(f"Gagal memproses gambar {image_path}: {str(e)}")
            return None

    def embed_batch(self, image_paths: List[str]) -> List[Optional[np.ndarray]]:
        """
        Mengubah batch gambar menjadi vektor embedding
        
        Args:
            image_paths: List path ke file gambar
        
        Returns:
            List numpy array atau None untuk gambar yang gagal diproses
        """
        embeddings = []
        for path in image_paths:
            embeddings.append(self.embed(path))
        return embeddings

    def save_embedding(self, image_path: str, output_path: Optional[str] = None) -> bool:
        """
        Menyimpan embedding gambar ke file .npy
        
        Args:
            image_path: Path ke file gambar
            output_path: Path untuk menyimpan embedding (default: folder gambar dengan ekstensi .npy)
        
        Returns:
            True jika berhasil, False jika gagal
        """
        embedding = self.embed(image_path)
        if embedding is None:
            return False
            
        if output_path is None:
            output_path = os.path.splitext(image_path)[0] + ".npy"
            
        try:
            np.save(output_path, embedding)
            logger.info(f"Embedding disimpan ke {output_path}")
            return True
        except Exception as e:
            logger.error(f"Gagal menyimpan embedding: {str(e)}")
            return False

# Contoh penggunaan
if __name__ == "__main__":
    # Test the embedder
    embedder = ImageEmbedder()
    
    # Contoh gambar (buat folder backend/data/images terlebih dahulu)
    test_image = os.path.join(settings.IMAGES_DIR, "test.jpg")
    
    # Pastikan folder exists
    os.makedirs(settings.IMAGES_DIR, exist_ok=True)
    
    # Jika ada gambar test, proses
    if os.path.exists(test_image):
        print("Memproses gambar test...")
        embedding = embedder.embed(test_image)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Sample embedding values: {embedding[:5]}")  # Cetak 5 nilai pertama
        
        # Test save embedding
        embedder.save_embedding(test_image)
    else:
        print(f"Buatlah file {test_image} untuk melakukan testing")