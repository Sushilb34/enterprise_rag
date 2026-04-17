import requests
import re
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
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.max_tokens = max_tokens
        self.temperature = temperature

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
                "stream": False
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

        response = requests.post(self.api_url, json=payload)
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