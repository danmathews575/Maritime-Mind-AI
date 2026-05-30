import base64
from openai import OpenAI
from app.configs.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.vision_ocr")

class NvidiaVisionService:
    def __init__(self):
        if not settings.NVIDIA_API_KEY:
            logger.warning("NVIDIA_API_KEY not set. Vision OCR will fail.")
            
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=settings.NVIDIA_API_KEY
        )
        self.model = "meta/llama-3.2-90b-vision-instruct"
        
    def extract_text_from_image(self, base64_img: str) -> str:
        """Extracts text from a base64 encoded image using Nvidia NIM Vision LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text from this page. Keep the structure as close to the original as possible. Only output the extracted text, no conversational filler or markdown code blocks."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                        ]
                    }
                ],
                max_tokens=2048,
                temperature=0.0,
                timeout=30.0
            )
            text = response.choices[0].message.content.strip()
            return text
        except Exception as e:
            logger.error(f"Vision OCR failed: {e}")
            return ""
