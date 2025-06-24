from backend.ai_core.memory_store import MemoryStore
from backend.ai_core.vector_store import VectorStore
from typing import Dict, Any
from datetime import datetime

class FeedbackSystem:
    def __init__(self):
        self.memory = MemoryStore()
        self.vector_store = VectorStore()
    
    def process_feedback(self, user_id: str, feedback_data: Dict[str, Any]):
        """Memproses feedback dengan lebih canggih"""
        # 1. Update user profile
        profile = self.memory.get_user_profile(user_id)
        
        # Initialize if not exists
        if 'feedback_stats' not in profile:
            profile['feedback_stats'] = {
                'positive': 0,
                'negative': 0,
                'last_updated': datetime.now().isoformat()
            }
        
        # Update stats
        if feedback_data.get('liked'):
            profile['feedback_stats']['positive'] += 1
        else:
            profile['feedback_stats']['negative'] += 1
        
        # 2. Learn from feedback content if provided
        if feedback_data.get('feedback_text'):
            self.vector_store.add_text(
                feedback_data['feedback_text'],
                {
                    'type': 'feedback',
                    'user_id': user_id,
                    'sentiment': 'positive' if feedback_data.get('liked') else 'negative'
                }
            )
        
        # 3. Update learning model preferences
        self._update_learning_model(user_id, feedback_data)
        
        # Save updated profile
        self.memory.update_user_profile(user_id, profile)
    
    def _update_learning_model(self, user_id: str, feedback_data: Dict[str, Any]):
        """Update model pembelajaran berdasarkan feedback"""
        # Implementasi bisa berupa:
        # - Menyesuaikan weights untuk rekomendasi
        # - Menandai topik yang disukai/tidak disukai
        # - Menyesuaikan gaya respons
        
        # Contoh sederhana: simpan preferensi gaya respons
        if feedback_data.get('preferred_style'):
            profile = self.memory.get_user_profile(user_id)
            preferred_styles = profile.get('preferred_styles', {})
            
            current_style = feedback_data['preferred_style']
            preferred_styles[current_style] = preferred_styles.get(current_style, 0) + 1
            
            profile['preferred_styles'] = preferred_styles
            self.memory.update_user_profile(user_id, profile)
    
    def get_improvement_suggestions(self, user_id: str):
        """Memberikan saran peningkatan berdasarkan feedback"""
        profile = self.memory.get_user_profile(user_id)
        
        # Dapatkan feedback negatif terbaru
        negative_feedback = list(self.memory.interactions.find({
            "metadata.type": "feedback",
            "metadata.sentiment": "negative",
            "metadata.user_id": user_id
        }).sort("timestamp", -1).limit(3))
        
        return {
            "profile": profile,
            "areas_for_improvement": negative_feedback
        }