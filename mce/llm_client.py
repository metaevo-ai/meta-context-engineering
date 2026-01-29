import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Callable, List, Union
from httpx._transports import default
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# DEFAULT_PROVIDER_CONFIG = {"quantizations": ["fp8"]}

class LLMClient:
    """Minimalist async LLM client with retry support."""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: float = 120.0,
        provider_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model name (e.g., "deepseek/deepseek-chat-v3.1")
            temperature: Temperature for sampling
            max_retries: Max retries for parsing failures and timeouts
            timeout: Timeout in seconds for each API call
            provider_config: OpenRouter provider config (e.g., {"only": ["openai"]})
        """
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENROUTER_API_BASE") or os.getenv("OPENAI_API_BASE")
        
        if not api_key:
            raise ValueError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        
        
        # if provider_config:
        #     provider_config.update(DEFAULT_PROVIDER_CONFIG)
        # else:
        #     provider_config = DEFAULT_PROVIDER_CONFIG
        self.provider_config = provider_config

    async def ainvoke(
        self,
        messages: Union[str, List[Dict[str, str]]],
        parse_function: Optional[Callable[[str], Any]] = None,
    ) -> Any:
        """
        Invoke LLM with optional parsing and retry.
        
        Args:
            messages: String prompt or list of message dicts with "role" and "content"
            parse_function: Optional function to parse response text
        
        Returns:
            Parsed output if parse_function provided, else raw text
        """
        # Convert string to messages format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        extra_body = {}
        if self.provider_config:
            extra_body["provider"] = self.provider_config
        
        for attempt in range(self.max_retries):
            try:
                # Wrap API call with timeout
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        extra_body=extra_body if extra_body else None,
                    ),
                    timeout=self.timeout
                )
                
                text = response.choices[0].message.content
                
                if parse_function:
                    return parse_function(text)
                return text
                
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} timed out after {self.timeout}s. Sleeping 10s before retrying...")
                    await asyncio.sleep(10)
                    continue
                logger.error(f"All {self.max_retries} attempts timed out")
                raise
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Sleeping 10s before retrying...")
                    await asyncio.sleep(10)
                    continue
                logger.error(f"All {self.max_retries} attempts failed")
                raise
        
        raise RuntimeError("Should not reach here")

