import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Settings:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.MONGO_DB_NAME = "jarvis_memory"

settings = Settings()
