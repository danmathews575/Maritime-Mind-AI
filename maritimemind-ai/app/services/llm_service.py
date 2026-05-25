import logging
import requests
import random
from typing import Optional
from app.configs.config import get_settings

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for integrating with LLM providers (Ollama, Gemini, OpenAI).
    Maintains a pool of instances for each available provider.
    """
    def __init__(self):
        self.settings = get_settings()
        self.pools = {}
        
        # 1. Initialize Ollama
        from langchain_community.llms import Ollama
        self.base_url = self.settings.OLLAMA_BASE_URL
        self.model_name = self.settings.OLLAMA_MODEL
        self.pools["ollama"] = [Ollama(
            base_url=self.base_url,
            model=self.model_name,
            temperature=0.0
        )]
        logger.info(f"Initialized Ollama LLM ({self.model_name})")
        
        # 2. Initialize Gemini (if keys exist)
        gemini_keys_str = getattr(self.settings, "GEMINI_API_KEY", "")
        if gemini_keys_str:
            from langchain_google_genai import ChatGoogleGenerativeAI
            keys = [k.strip() for k in gemini_keys_str.split(",") if k.strip()]
            self.pools["gemini"] = []
            for key in keys:
                self.pools["gemini"].append(ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=key,
                    temperature=0.0
                ))
            logger.info(f"Initialized Gemini pool with {len(keys)} keys")
            
        # 3. Initialize OpenAI (if keys exist)
        openai_keys_str = getattr(self.settings, "OPENAI_API_KEY", "")
        if openai_keys_str:
            from langchain_openai import ChatOpenAI
            keys = [k.strip() for k in openai_keys_str.split(",") if k.strip()]
            self.pools["openai"] = []
            for key in keys:
                self.pools["openai"].append(ChatOpenAI(
                    model="gpt-4o",
                    api_key=key,
                    temperature=0.0
                ))
            logger.info(f"Initialized OpenAI pool with {len(keys)} keys")

    def _get_llm(self, provider: str):
        """Randomly selects an LLM from the specified provider pool."""
        provider = provider.lower() if provider else "ollama"
        if provider not in self.pools or not self.pools[provider]:
            logger.warning(f"Provider '{provider}' not available. Falling back to ollama.")
            provider = "ollama"
        return random.choice(self.pools[provider]), provider

    def health_check(self) -> bool:
        """
        Verify that the default Ollama service is running.
        Remote APIs are assumed healthy if keys are present.
        """
        try:
            response = requests.get(self.base_url)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, provider: Optional[str] = None) -> str:
        """
        Generate a response using the requested LLM provider.
        """
        llm, active_provider = self._get_llm(provider)
        
        if active_provider == "ollama":
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            try:
                response = llm.invoke(full_prompt)
                return response
            except Exception as e:
                logger.error(f"Ollama generation failed: {e}")
                return f"Error generating response: {str(e)}"
        else:
            # For Chat Models (Gemini/OpenAI)
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            try:
                response = llm.invoke(messages)
                return response.content
            except Exception as e:
                logger.error(f"{active_provider} generation failed: {e}")
                return f"Error generating response: {str(e)}"
