import requests
import re
from urllib.parse import urlparse
from app.core.config import get_settings
from app.core.logger import get_logger
logger = get_logger()
settings = get_settings()

class LocalLLMClient:
    def __init__(
        self,
        model_name: str = settings.LOCAL_LLM_MODEL,
        api_url: str = settings.LOCAL_LLM_API_URL,
        max_tokens: int = settings.LOCAL_LLM_MAX_TOKENS,
        temperature: float = settings.LOCAL_LLM_TEMPERATURE,
        connect_timeout: float = settings.LOCAL_LLM_CONNECT_TIMEOUT,
        read_timeout: float = settings.LOCAL_LLM_READ_TIMEOUT,
        enable_thinking: bool = settings.LOCAL_LLM_ENABLE_THINKING,
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        # (connect, read) timeout tuple for requests; prevents a hung backend
        # from blocking the worker thread forever.
        self.timeout = (connect_timeout, read_timeout)

    def health_check(self) -> bool:
        """
        Ping the vLLM server's /health endpoint (derived from the API URL).
        Returns True only if it responds 200 within a short timeout. Used by
        the /health/ready readiness probe — kept cheap and fast on purpose.
        """
        parsed = urlparse(self.api_url)
        health_url = f"{parsed.scheme}://{parsed.netloc}/health"
        try:
            resp = requests.get(health_url, timeout=(3, 5))
            return resp.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"LLM backend health check failed for {health_url}: {e}")
            return False

    def _clean_response(self, text: str) -> str:
        """
        Remove <think> blocks from LLM response.
        Handles:
        - Multiple <think> blocks
        - Unclosed <think> tags (matches until end of string)
        - Multi-line content
        """
        if not text:
            return ""
        
        # Regex to match <think> and everything until </think> or end of string
        # Non-greedy .*? ensures we match blocks one by one
        pattern = r'<think>.*?(?:</think>|$)'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned.strip()

    def _truncate_prompt(self, prompt: str, max_context: int = 16384) -> str:
        """Truncate prompt to fit within the model's max context length."""
        # Rough estimate: 1 token ≈ 4 characters
        # Ensure available_context is positive
        available_context = max(500, max_context - self.max_tokens - 100)
        max_chars = available_context * 4
        
        if len(prompt) > max_chars:
            logger.warning(f"Prompt truncated from {len(prompt)} to {max_chars} chars to fit context window")
            prompt = prompt[:max_chars]
        return prompt

    def generate(self, prompt: str, stop: list = None) -> str:
        prompt = self._truncate_prompt(prompt)
        
        # Detect endpoint type
        is_chat = "chat/completions" in self.api_url
        
        if is_chat:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False,
                # vLLM passes this to the Qwen3 chat template. Disabling
                # thinking skips the (discarded) <think> reasoning tokens,
                # cutting latency ~3x. See LOCAL_LLM_ENABLE_THINKING.
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            }
        else:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
        
        if stop:
            payload["stop"] = stop

        response = requests.post(self.api_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        text = ""

        # Handle Chat vs Completion response formats
        if is_chat:
            if "choices" in data and len(data["choices"]) > 0:
                text = data["choices"][0].get("message", {}).get("content", "")
        else:
            if "choices" in data and len(data["choices"]) > 0:
                text = data["choices"][0].get("text", "")
        
        # If still empty, try fallback
        if not text:
            text = data.get("response", "")

        # Clean response (remove <think> tags) before returning
        return self._clean_response(text)