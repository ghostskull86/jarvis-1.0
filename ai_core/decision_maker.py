from backend.ai_core.gemini_api import GeminiAPI
from backend.ai_core.memory_store import MemoryStore
from typing import Dict, Any

class DecisionMaker:
    def __init__(self):
        self.gemini = GeminiAPI()
        self.memory = MemoryStore()
    
    def make_decision(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Membuat keputusan berdasarkan konteks dan riwayat pengguna"""
        # Dapatkan riwayat pengguna
        history = self.memory.get_interaction_history(user_id=user_id, limit=5)
        profile = self.memory.get_user_profile(user_id)
        
        # Format prompt untuk pengambilan keputusan
        prompt = f"""
        Anda adalah asisten AI yang membantu pengguna membuat keputusan.
        Profil pengguna: {profile}
        Riwayat terakhir: {history}
        
        Konteks saat ini:
        {context}
        
        Berikan rekomendasi keputusan dengan struktur:
        - Analisis situasi
        - Opsi yang tersedia
        - Rekomendasi terbaik
        - Alasan
        """
        
        decision = self.gemini.generate_text(prompt)
        
        # Simpan keputusan yang dibuat
        self.memory.log_interaction(
            user_input=str(context),
            ai_response=decision,
            metadata={
                "type": "decision",
                "user_id": user_id
            }
        )
        
        return {
            "decision": decision,
            "context": context
        }