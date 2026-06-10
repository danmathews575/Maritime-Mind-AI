from app.utils.logger import setup_logger
import requests
import random
import time
from typing import Optional, List, Tuple
from app.configs.config import get_settings

logger = setup_logger("maritimemind.services.llm")


class LLMService:
    """
    Service for integrating with LLM providers with automatic fallback.

    Fallback Chain (configurable via LLM_FALLBACK_ORDER):
        1. NVIDIA NIM API  — primary cloud inference
        2. Google Gemini   — secondary cloud fallback
        3. Ollama (local)  — tertiary local fallback

    On each generate() call, the service tries providers in order.
    If a provider fails (network error, rate limit, timeout, etc.),
    it automatically falls through to the next provider in the chain.
    """

    def __init__(self):
        self.settings = get_settings()
        self.pools: dict[str, list] = {}
        self._init_errors: dict[str, str] = {}

        # Parse the fallback order from settings
        self.fallback_order: List[str] = [
            p.strip().lower()
            for p in self.settings.LLM_FALLBACK_ORDER.split(",")
            if p.strip()
        ]

        # Initialize each provider gracefully — failures are logged, not raised
        self._init_nvidia()
        self._init_gemini()
        self._init_ollama()
        self._init_openai()

        # Report initialization summary
        available = [p for p in self.fallback_order if p in self.pools]
        unavailable = [p for p in self.fallback_order if p not in self.pools]
        logger.info(
            f"LLM Fallback Chain initialized. "
            f"Order: {' -> '.join(self.fallback_order)} | "
            f"Available: {available} | Unavailable: {unavailable}"
        )
        if not available:
            logger.error(
                "No LLM providers are available! All providers failed to initialize. "
                f"Errors: {self._init_errors}"
            )

    # ── Provider Initialization ──────────────────────────────────────────────

    def _init_nvidia(self):
        """Initialize NVIDIA NIM API provider."""
        nvidia_keys_str = getattr(self.settings, "NVIDIA_API_KEY", "")
        if not nvidia_keys_str:
            self._init_errors["nvidia"] = "No API key configured"
            return
        try:
            from langchain_openai import ChatOpenAI
            keys = [k.strip() for k in nvidia_keys_str.split(",") if k.strip()]
            self.pools["nvidia"] = []
            for key in keys:
                self.pools["nvidia"].append(ChatOpenAI(
                    model="meta/llama-3.1-70b-instruct",
                    api_key=key,
                    base_url="https://integrate.api.nvidia.com/v1",
                    temperature=0.0,
                    request_timeout=30,
                ))
            logger.info(f"[OK] NVIDIA NIM initialized ({len(keys)} key(s))")
        except Exception as e:
            self._init_errors["nvidia"] = str(e)
            logger.warning(f"⚠️ NVIDIA NIM initialization failed: {e}")

    def _init_gemini(self):
        """Initialize Google Gemini provider."""
        gemini_keys_str = getattr(self.settings, "GEMINI_API_KEY", "")
        if not gemini_keys_str:
            self._init_errors["gemini"] = "No API key configured"
            return
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            keys = [k.strip() for k in gemini_keys_str.split(",") if k.strip()]
            self.pools["gemini"] = []
            for key in keys:
                self.pools["gemini"].append(ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=key,
                    temperature=0.0,
                    request_timeout=30,
                ))
            logger.info(f"[OK] Gemini initialized ({len(keys)} key(s))")
        except Exception as e:
            self._init_errors["gemini"] = str(e)
            logger.warning(f"⚠️ Gemini initialization failed: {e}")

    def _init_ollama(self):
        """Initialize Ollama local provider."""
        try:
            from langchain_community.llms import Ollama
            self.pools["ollama"] = [Ollama(
                base_url=self.settings.OLLAMA_BASE_URL,
                model=self.settings.OLLAMA_MODEL,
                temperature=0.0,
            )]
            logger.info(f"[OK] Ollama initialized ({self.settings.OLLAMA_MODEL})")
        except Exception as e:
            self._init_errors["ollama"] = str(e)
            logger.warning(f"⚠️ Ollama initialization failed: {e}")

    def _init_openai(self):
        """Initialize OpenAI provider (bonus, not in default chain)."""
        openai_keys_str = getattr(self.settings, "OPENAI_API_KEY", "")
        if not openai_keys_str:
            return
        try:
            from langchain_openai import ChatOpenAI
            keys = [k.strip() for k in openai_keys_str.split(",") if k.strip()]
            self.pools["openai"] = []
            for key in keys:
                self.pools["openai"].append(ChatOpenAI(
                    model="gpt-4o",
                    api_key=key,
                    temperature=0.0,
                    request_timeout=30,
                ))
            logger.info(f"[OK] OpenAI initialized ({len(keys)} key(s))")
        except Exception as e:
            self._init_errors["openai"] = str(e)
            logger.warning(f"⚠️ OpenAI initialization failed: {e}")

    # ── Provider Selection ───────────────────────────────────────────────────

    def _get_fallback_chain(self, preferred_provider: Optional[str] = None) -> List[str]:
        """
        Build the ordered list of providers to try.

        If a preferred_provider is specified, it goes first,
        followed by the remaining fallback order.
        """
        if preferred_provider:
            preferred = preferred_provider.lower()
            chain = [preferred]
            for p in self.fallback_order:
                if p != preferred:
                    chain.append(p)
            return chain
        return list(self.fallback_order)

    def _pick_from_pool(self, provider: str):
        """Randomly select an LLM instance from a provider's pool."""
        if provider in self.pools and self.pools[provider]:
            return random.choice(self.pools[provider])
        return None

    # ── Health Check ─────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """
        Check health of all configured providers.
        Returns a dict of provider → status.
        """
        status = {}

        # Ollama: actual connectivity check
        if "ollama" in self.pools:
            try:
                response = requests.get(
                    self.settings.OLLAMA_BASE_URL, timeout=5
                )
                status["ollama"] = response.status_code == 200
            except requests.exceptions.ConnectionError:
                status["ollama"] = False
        else:
            status["ollama"] = False

        # Cloud APIs: considered healthy if keys are present and init succeeded
        for provider in ["nvidia", "gemini", "openai"]:
            status[provider] = provider in self.pools and len(self.pools[provider]) > 0

        return status

    def health_check_legacy(self) -> bool:
        """Legacy health check — returns True if ANY provider is available."""
        statuses = self.health_check()
        return any(statuses.values())

    # ── Provider Info ────────────────────────────────────────────────────────

    def get_provider_status(self) -> dict:
        """
        Returns detailed provider status for diagnostics.
        """
        return {
            "fallback_order": self.fallback_order,
            "available_providers": [p for p in self.fallback_order if p in self.pools],
            "unavailable_providers": {
                p: self._init_errors.get(p, "unknown")
                for p in self.fallback_order
                if p not in self.pools
            },
            "pool_sizes": {p: len(instances) for p, instances in self.pools.items()},
        }

    # ── Core Generation with Fallback ────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> str:
        """
        Generate a response using the LLM fallback chain.

        Tries each provider in the configured order. If a provider fails
        (network error, rate limit, timeout, API error), automatically
        falls through to the next provider.

        Args:
            prompt: The user/task prompt.
            system_prompt: Optional system instruction.
            provider: Force a specific starting provider (still falls back on failure).

        Returns:
            The generated text response.
        """
        chain = self._get_fallback_chain(provider)
        errors: List[Tuple[str, str]] = []

        for attempt_provider in chain:
            llm = self._pick_from_pool(attempt_provider)
            if llm is None:
                errors.append((
                    attempt_provider,
                    self._init_errors.get(attempt_provider, "Provider not initialized")
                ))
                continue

            try:
                start_time = time.time()
                result = self._invoke_provider(llm, attempt_provider, prompt, system_prompt)
                elapsed = time.time() - start_time

                # Log success (and whether fallback was used)
                if errors:
                    failed_names = [e[0] for e in errors]
                    logger.warning(
                        f"[RETRY] Fallback activated: {' -> '.join(failed_names)} failed. "
                        f"Succeeded with '{attempt_provider}' in {elapsed:.1f}s"
                    )
                else:
                    logger.info(
                        f"LLM response from '{attempt_provider}' in {elapsed:.1f}s"
                    )
                return result

            except Exception as e:
                error_msg = str(e)
                errors.append((attempt_provider, error_msg))
                logger.warning(
                    f"🔄 Provider '{attempt_provider}' failed: {error_msg}. "
                    f"Trying next in fallback chain..."
                )
                continue

        # All providers exhausted
        error_summary = "; ".join(f"{p}: {e}" for p, e in errors)
        logger.error(f"[ERROR] All LLM providers failed. Errors: {error_summary}")
        return (
            "I'm temporarily unable to generate a response — all LLM providers "
            "are currently unavailable. Please try again in a moment.\n\n"
            f"**Provider errors**: {error_summary}"
        )

    def _invoke_provider(
        self, llm, provider: str, prompt: str, system_prompt: Optional[str]
    ) -> str:
        """
        Invoke a specific LLM provider instance.

        Ollama uses plain text invocation; Chat-based models (Gemini, OpenAI,
        NVIDIA) use message-based invocation with SystemMessage/HumanMessage.
        """
        if provider == "ollama":
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = llm.invoke(full_prompt)
            return response
        else:
            # Chat-based models (Gemini, OpenAI, NVIDIA NIM)
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            response = llm.invoke(messages)
            return response.content
