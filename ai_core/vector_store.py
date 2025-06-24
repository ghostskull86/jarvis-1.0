import faiss
import numpy as np
from backend.config import settings
import os
from backend.ai_core.embedder import TextEmbedder

class VectorStore:
    def __init__(self):
        self.embedder = TextEmbedder()
        self.index_path = settings.FAISS_INDEX_DIR / "main_index.faiss"
        self.dimension = 384  # Sesuai dengan model all-MiniLM-L6-v2
        
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def add_text(self, text: str, metadata: dict):
        """Menambahkan teks ke vector store"""
        embedding = self.embedder.embed(text)
        vector = np.array([embedding], dtype='float32')
        
        # Untuk metadata, kita perlu menyimpan secara terpisah
        # Di implementasi nyata, Anda perlu database tambahan untuk metadata
        self.index.add(vector)
        faiss.write_index(self.index, str(self.index_path))
    
    def search(self, query: str, k: int = 3):
        """Mencari teks yang mirip"""
        embedding = self.embedder.embed(query)
        vector = np.array([embedding], dtype='float32')
        
        distances, indices = self.index.search(vector, k)
        
        # Di implementasi nyata, Anda akan mengambil metadata dari database
        return [{"id": int(idx), "score": float(score)} 
                for idx, score in zip(indices[0], distances[0])]