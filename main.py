from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import uuid
import sys
from pathlib import Path


# Import modul dari ai_core
from ai_core.gemini_api import GeminiAPI
from ai_core.memory_store import MemoryStore
from config import settings

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Jarvis AI Assistant",
    description="Personal AI Assistant dengan kemampuan multimodal",
    version="0.1.0"
)

# Setup CORS (untuk development saja)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi komponen AI
gemini = GeminiAPI()
memory = MemoryStore()

@app.get("/")
async def root():
    return {
        "message": "Jarvis AI Assistant API",
        "status": "running",
        "version": "0.1.0"
    }

@app.post("/chat")
async def chat(message: str, user_id: str = "default"):
    """Endpoint untuk berinteraksi dengan AI"""
    try:
        # Generate response menggunakan Gemini
        response = gemini.generate_text(message)
        
        # Simpan interaksi ke memory
        memory.log_interaction(
            user_input=message,
            ai_response=response,
            metadata={"user_id": user_id}
        )
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)