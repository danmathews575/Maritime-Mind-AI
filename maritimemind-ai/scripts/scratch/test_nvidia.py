import os
import base64
from openai import OpenAI
from app.configs.config import settings

def test_nvidia_vision():
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.NVIDIA_API_KEY
    )
    
    response = client.chat.completions.create(
        model="meta/llama-3.2-90b-vision-instruct",
        messages=[
            {
                "role": "user",
                "content": "What is 2+2?"
            }
        ],
        max_tokens=1024
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    test_nvidia_vision()
