from backend.ai_core.memory_store import MemoryStore
from backend.ai_core.gemini_api import GeminiAPI
from typing import List, Dict

class PersonalityAnalyzer:
    def __init__(self):
        self.memory = MemoryStore()
        self.gemini = GeminiAPI()
    
    def analyze_from_history(self, user_id: str) -> Dict[str, str]:
        """Menganalisis kepribadian berdasarkan riwayat chat"""
        history = self.memory.get_interaction_history(user_id=user_id, limit=50)
        
        if not history:
            return {"error": "Tidak ada data yang cukup"}
        
        # Format riwayat untuk analisis
        formatted_history = "\n".join(
            f"User: {item['user_input']}\nAI: {item['ai_response']}"
            for item in history
        )
        
        prompt = f"""
        Berdasarkan riwayat percakapan berikut, analisis kepribadian pengguna:
        - Gaya komunikasi
        - Minat/topik yang sering dibicarakan
        - Kemungkinan sifat kepribadian
        
        Berikan dalam format JSON dengan keys: communication_style, interests, personality_traits
        
        Riwayat percakapan:
        {formatted_history}
        """
        
        analysis = self.gemini.generate_text(prompt)
        return self._parse_analysis(analysis)
    
    def _parse_analysis(self, text: str) -> Dict[str, str]:
        """Mengubah teks analisis menjadi struktur data"""
        # Ini implementasi sederhana, bisa diperbaiki dengan parsing yang lebih baik
        try:
            # Asumsi Gemini mengembalikan format JSON yang valid
            import json
            return json.loads(text.strip())
        except:
            return {"raw_analysis": text}