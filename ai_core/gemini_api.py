import google.generativeai as genai
from config.settings import settings
from typing import Optional
import PIL.Image  # dipindahkan ke atas untuk konsistensi

class GeminiAPI:
    print("DEBUG: settings.GEMINI_API_KEY =", settings.GEMINI_API_KEY)

    def __init__(self):
        print("DEBUG: GEMINI API KEY:", settings.GEMINI_API_KEY)
        genai.configure(api_key=settings.gemini_api_key)

        self.model = genai.getGenerativeModel("gemini-2.0-flash")
        self.vision_model = genai.getGenerativeModel("gemini-2.0-flash")
    def generate_text(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate text response using Gemini"""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = self.model.generate_content(full_prompt)
        return response.text   
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """Analyze image using Gemini Vision"""
        img = PIL.Image.open(image_path)
        response = self.vision_model.generate_content([prompt, img])
        return response.text
