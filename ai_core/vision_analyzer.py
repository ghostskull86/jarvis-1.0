from backend.ai_core.gemini_api import GeminiAPI
from backend.ai_core.image_embedder import ImageEmbedder
from typing import Optional

class VisionAnalyzer:
    def __init__(self):
        self.gemini = GeminiAPI()
        self.embedder = ImageEmbedder()
    
    def analyze(self, image_path: str, prompt: Optional[str] = None):
        """Analisis gambar dengan kombinasi Gemini dan embedding"""
        # Analisis menggunakan Gemini
        gemini_response = self.gemini.analyze_image(
            image_path,
            prompt or "Jelaskan gambar ini secara detail"
        )
        
        # Dapatkan embedding gambar
        image_embedding = self.embedder.embed(image_path)
        
        return {
            "description": gemini_response,
            "embedding": image_embedding.tolist()  # Convert numpy array ke list
        }