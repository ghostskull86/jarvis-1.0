from pymongo import MongoClient
from datetime import datetime
from config import settings
from typing import Dict, Any

class MemoryStore:
    def __init__(self):
        self.client = MongoClient(settings.MONGO_URI)
        self.db = self.client[settings.MONGO_DB_NAME]
        self.interactions = self.db["interactions"]
        self.user_profiles = self.db["user_profiles"]
    
    def log_interaction(self, user_input: str, ai_response: str, metadata: Dict[str, Any] = None):
        """Menyimpan interaksi pengguna dengan AI"""
        record = {
            "timestamp": datetime.now(),
            "user_input": user_input,
            "ai_response": ai_response,
            "metadata": metadata or {}
        }
        self.interactions.insert_one(record)
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Mengambil profil pengguna"""
        return self.user_profiles.find_one({"user_id": user_id}) or {}
    
    def update_user_profile(self, user_id: str, updates: Dict[str, Any]):
        """Memperbarui profil pengguna"""
        self.user_profiles.update_one(
            {"user_id": user_id},
            {"$set": updates},
            upsert=True
        )
    
    def get_interaction_history(self, limit: int = 10):
        """Mengambil riwayat interaksi terakhir"""
        return list(self.interactions.find().sort("timestamp", -1).limit(limit))