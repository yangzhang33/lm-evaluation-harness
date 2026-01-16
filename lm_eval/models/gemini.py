"""
Gemini API model wrapper for lm-evaluation-harness.

Uses the google-genai SDK to communicate with Google's Gemini models.
Supports Gemini 3 Pro, Gemini 3 Flash, and other Gemini models.

Usage:
    lm_eval --model gemini --model_args model=gemini-3-pro-preview --tasks <task> --apply_chat_template
"""

import logging
import os
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences


eval_logger = logging.getLogger(__name__)


@register_model("gemini", "gemini-chat")
class GeminiChatCompletion(TemplateAPI):
    """
    Gemini API model wrapper using the google-genai SDK.
    
    Supports both Gemini 3 series and older Gemini models.
    
    Note:
        - tokenizer_backend is accepted but ignored (Gemini handles tokenization internally)
        - stop_sequences are passed to Gemini but may not always be honored exactly
        - Gemini does not support loglikelihood/logprobs
    """
    
    def __init__(
        self,
        model: str = "gemini-3-pro-preview",
        tokenizer_backend: str = None,
        tokenized_requests: bool = False,
        thinking_level: str = None,  # "low", "medium", "high" for Gemini 3
        **kwargs,
    ):
        eval_logger.warning(
            "Gemini chat-completions requires the `--apply_chat_template` flag."
        )
        
        self.thinking_level = thinking_level
        self._client = None
        
        # Gemini doesn't use base_url in the same way as OpenAI
        # We'll handle API calls through the SDK
        super().__init__(
            model=model,
            base_url="https://generativelanguage.googleapis.com",  # placeholder
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        
        if self._batch_size > 1:
            eval_logger.warning(
                "Gemini does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1
    
    @cached_property
    def client(self):
        """Initialize and return the Gemini client."""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai is not installed. Please install it with: pip install google-genai"
            )
        
        api_key = self.api_key
        return genai.Client(api_key=api_key)
    
    @cached_property
    def api_key(self):
        """Return the API key for Gemini."""
        key = os.environ.get("GOOGLE_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `GOOGLE_API_KEY` environment variable."
            )
        return key
    
    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        """Create the payload for Gemini API - not used directly, see model_call."""
        assert type(messages) is not str, (
            "Gemini chat-completions require the --apply_chat_template flag."
        )
        
        if gen_kwargs is None:
            gen_kwargs = {}
        
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        
        temperature = gen_kwargs.pop("temperature", 1.0)  # Gemini 3 recommends 1.0
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        
        return {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
            **gen_kwargs,
        }
    
    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[Dict]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Override model_call to use the Gemini SDK directly instead of HTTP requests.
        """
        from google.genai import types
        
        if gen_kwargs is None:
            gen_kwargs = {}
        
        # Parse generation kwargs
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        
        # FIX: Gemini 3 "Thinking" models require more tokens for thoughts
        # Task configs often set max_tokens=10 for MCQA, which causes immediate truncation
        if "3" in self.model:
            max_tokens = max(max_tokens, 8192)  # Ensure enough budget to think
        
        temperature = gen_kwargs.pop("temperature", 1.0)
        stop_sequences = gen_kwargs.pop("until", None)
        if stop_sequences and not isinstance(stop_sequences, list):
            stop_sequences = [stop_sequences]
        
        # Convert messages to Gemini format
        contents = self._convert_messages_to_gemini_format(messages)
        
        # Build generation config
        config_dict = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if stop_sequences:
            # Filter out None and empty strings
            stop_sequences = [s for s in stop_sequences if s]
            if stop_sequences:
                config_dict["stop_sequences"] = stop_sequences[:5]  # Gemini max is 5
        
        # Add thinking level for Gemini 3 models (wrapped defensively)
        if self.thinking_level and "3" in self.model:
            try:
                config_dict["thinking_config"] = types.ThinkingConfig(
                    thinking_level=self.thinking_level
                )
            except Exception as e:
                eval_logger.warning(f"Could not set thinking_config: {e}")
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_dict),
            )
            
            # DEBUG: Log the full response to see why text is empty
            if not response.text:
                eval_logger.warning(f"Empty response text. Full object: {response}")
            
            # Extract text safely - response.text is not always populated
            text = ""
            if response.text:
                text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                try:
                    parts = response.candidates[0].content.parts
                    text = "".join(p.text for p in parts if hasattr(p, "text"))
                except Exception:
                    pass
            
            # Return in a format compatible with parse_generations
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": text
                        }
                    }
                ]
            }
        except Exception as e:
            eval_logger.error(f"Gemini API error: {e}")
            # Return empty response on error
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": ""
                        }
                    }
                ]
            }
    
    def _convert_messages_to_gemini_format(self, messages):
        """
        Convert OpenAI-style messages to Gemini format.
        
        OpenAI format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        Gemini format: "string" or [{"role": "user", "parts": [{"text": "..."}]}]
        """
        if isinstance(messages, str):
            return messages
        
        if isinstance(messages, list) and len(messages) > 0:
            # If it's a list of dicts with role/content (OpenAI format)
            if isinstance(messages[0], dict) and "role" in messages[0]:
                gemini_contents = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    # Map roles: OpenAI uses "assistant", Gemini uses "model"
                    if role == "assistant":
                        role = "model"
                    elif role == "system":
                        # Prepend system message to first user message
                        # or treat as user message
                        role = "user"
                    
                    # Handle content that might be a list (multimodal)
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                parts.append({"text": item["text"]})
                            elif isinstance(item, str):
                                parts.append({"text": item})
                        gemini_contents.append({"role": role, "parts": parts})
                    else:
                        gemini_contents.append({
                            "role": role,
                            "parts": [{"text": str(content)}]
                        })
                
                return gemini_contents
            
            # If it's already in a simple format, just return as string
            if isinstance(messages[0], str):
                return "\n".join(messages)
        
        return str(messages)
    
    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """Parse the generations from Gemini response."""
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out.get("choices", []))
            for choice in out.get("choices", []):
                idx = choice.get("index", 0)
                content = choice.get("message", {}).get("content", "")
                tmp[idx] = content
            res = res + tmp
        return res if res else [""]

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Logprobs are not supported by Gemini.
        Returns empty list to satisfy abstract method requirement.
        """
        return []
    
    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        """Gemini handles tokenization internally."""
        return string
    
    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for Gemini. "
            "Use generate_until tasks instead."
        )
