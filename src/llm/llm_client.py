#!/usr/bin/env python3
"""
LLM Client for Coaching and Analysis
====================================

This module provides a unified interface for different LLM providers
to generate intelligent coaching feedback and analysis.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    success: bool
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None

class LLMClient:
    """Unified LLM client supporting multiple providers"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM client with configuration"""
        self.config = config
        self.provider = config.get('provider', 'ollama')
        self.model_name = config.get('model_name', 'llama2')
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.api_key = config.get('api_key')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        self.timeout = config.get('timeout_seconds', 60)
        self.max_retries = config.get('max_retries', 3)
        
        logger.info(f"LLM Client initialized with {self.provider}:{self.model_name}")
    
    def generate_coaching_feedback(self, prompt: str) -> LLMResponse:
        """Generate coaching feedback using the configured LLM"""
        try:
            start_time = time.time()
            
            if self.provider == 'ollama':
                response = self._call_ollama(prompt)
            elif self.provider == 'openai':
                response = self._call_openai(prompt)
            elif self.provider == 'anthropic':
                response = self._call_anthropic(prompt)
            else:
                return LLMResponse(
                    content="",
                    success=False,
                    error_message=f"Unsupported LLM provider: {self.provider}"
                )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.get('content', ''),
                success=True,
                tokens_used=response.get('tokens_used'),
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Error generating coaching feedback: {e}")
            return LLMResponse(
                content="",
                success=False,
                error_message=str(e)
            )
    
    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API"""
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                'content': result.get('response', ''),
                'tokens_used': result.get('eval_count', 0)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(f"Failed to call Ollama: {e}")
    
    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API"""
        if not self.api_key:
            raise Exception("OpenAI API key required")
        
        try:
            url = "https://api.openai.com/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert interview coach specializing in technical interviews."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                'content': result['choices'][0]['message']['content'],
                'tokens_used': result['usage']['total_tokens']
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"Failed to call OpenAI: {e}")
    
    def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic Claude API"""
        if not self.api_key:
            raise Exception("Anthropic API key required")
        
        try:
            url = "https://api.anthropic.com/v1/messages"
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature
            }
            
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                'content': result['content'][0]['text'],
                'tokens_used': result['usage']['input_tokens'] + result['usage']['output_tokens']
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Anthropic API error: {e}")
            raise Exception(f"Failed to call Anthropic: {e}")
    
    def test_connection(self) -> bool:
        """Test if the LLM provider is accessible"""
        try:
            if self.provider == 'ollama':
                response = requests.get(f"{self.base_url}/api/tags", timeout=10)
                return response.status_code == 200
            else:
                # For other providers, we'd need to implement specific health checks
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
