"""
LLM Client for Qwen API
"""

import json
import re
from typing import Dict, Optional

from utils.config import DASHSCOPE_API_KEY, DASHSCOPE_API_BASE, QWEN_MODEL


class QwenClient:
    """Wrapper for Qwen API (supports both dashscope and openai-compatible mode)"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = QWEN_MODEL):
        self.api_key = api_key or DASHSCOPE_API_KEY
        self.base_url = base_url or DASHSCOPE_API_BASE
        self.model = model
        self.client = None
        self.use_dashscope = False
        
        # Token usage tracking
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        
        # Try openai-compatible mode first
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            # Fall back to dashscope native
            self.use_dashscope = True
    
    def reset_token_count(self):
        """Reset token counters (call before each case synthesis)."""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
    
    def get_token_count(self) -> Dict:
        """Return accumulated token usage since last reset."""
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens,
        }
    
    def generate(self, messages: list, model: str = None, enable_thinking: bool = False, **kwargs) -> tuple:
        """Call LLM with messages list (streaming) and return (reasoning_content, answer_content, usage)."""
        if self.use_dashscope:
            raise NotImplementedError("generate() requires openai-compatible mode; dashscope fallback not supported")

        use_model = model or self.model
        completion = self.client.chat.completions.create(
            model=use_model,
            messages=messages,
            extra_body={"enable_thinking": enable_thinking},
            stream=True,
            stream_options={"include_usage": True},
            **kwargs
        )

        reasoning_content = ""
        answer_content = ""
        usage = None
        for chunk in completion:
            if not chunk.choices:
                usage = chunk.usage
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content
            if hasattr(delta, "content") and delta.content:
                answer_content += delta.content

        if usage:
            self._prompt_tokens += usage.prompt_tokens or 0
            self._completion_tokens += usage.completion_tokens or 0
            self._total_tokens += usage.total_tokens or 0

        return reasoning_content, answer_content, usage

    def call(self, prompt: str, temperature: float = 0.7) -> str:
        """Call LLM and return response text"""
        try:
            if self.use_dashscope:
                return self._call_dashscope(prompt, temperature)
            else:
                return self._call_openai(prompt, temperature)
        except Exception as e:
            print(f"Exception calling Qwen API: {e}")
            return ""
    
    def _call_openai(self, prompt: str, temperature: float) -> str:
        """Call using openai-compatible API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        if response.usage:
            self._prompt_tokens += response.usage.prompt_tokens or 0
            self._completion_tokens += response.usage.completion_tokens or 0
            self._total_tokens += response.usage.total_tokens or 0
        return response.choices[0].message.content
    
    def _call_dashscope(self, prompt: str, temperature: float) -> str:
        """Call using dashscope native API"""
        from dashscope import Generation
        
        response = Generation.call(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            api_key=self.api_key
        )
        
        if response.status_code == 200:
            if hasattr(response, 'usage') and response.usage:
                self._prompt_tokens += getattr(response.usage, 'input_tokens', 0)
                self._completion_tokens += getattr(response.usage, 'output_tokens', 0)
                self._total_tokens += (
                    getattr(response.usage, 'input_tokens', 0)
                    + getattr(response.usage, 'output_tokens', 0)
                )
            return response.output.text
        else:
            print(f"Dashscope error: {response.code} - {response.message}")
            return ""
    
    def call_json(self, prompt: str, temperature: float = 0.7) -> Optional[Dict]:
        """Call LLM and parse JSON response"""
        response = self.call(prompt, temperature)
        if not response:
            return None
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON object
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        return None
