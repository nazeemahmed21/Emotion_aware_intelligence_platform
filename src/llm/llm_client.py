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
            # First, check if the model is available
            self._validate_ollama_model()
            
            url = f"{self.base_url}/api/generate"
            
            # Truncate prompt if it's too long (Ollama has limits)
            max_prompt_length = 4000  # Conservative limit
            if len(prompt) > max_prompt_length:
                logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to {max_prompt_length}")
                prompt = prompt[:max_prompt_length] + "\n\n[Content truncated due to length]"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": min(self.max_tokens, 1000)  # Conservative token limit
                }
            }
            
            logger.debug(f"Calling Ollama with model: {self.model_name}, prompt length: {len(prompt)}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 500:
                # Try to get more detailed error information
                try:
                    error_detail = response.json()
                    logger.error(f"Ollama 500 error details: {error_detail}")
                    
                    # Check if it's a CUDA memory error
                    if 'CUDA' in str(error_detail) and 'buffer' in str(error_detail):
                        logger.info("CUDA memory error detected, trying CPU fallback...")
                        return self._call_ollama_cpu_fallback(prompt)
                        
                except:
                    logger.error("Ollama 500 error - no additional details available")
                
                # Try with a simpler prompt as fallback
                return self._call_ollama_fallback(prompt)
            
            response.raise_for_status()
            
            result = response.json()
            return {
                'content': result.get('response', ''),
                'tokens_used': result.get('eval_count', 0)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(f"Failed to call Ollama: {e}")
    
    def _call_ollama_cpu_fallback(self, prompt: str) -> Dict[str, Any]:
        """Fallback Ollama call using CPU instead of GPU"""
        try:
            # Create a much simpler prompt
            simplified_prompt = self._simplify_prompt(prompt)
            
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": simplified_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,  # Lower temperature for more stable output
                    "num_predict": 500,   # Shorter response
                    "num_gpu": 0          # Force CPU usage
                }
            }
            
            logger.info("Attempting Ollama call with CPU fallback")
            
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
            
        except Exception as e:
            logger.error(f"Ollama CPU fallback also failed: {e}")
            # Return a basic response
            return {
                'content': "I apologize, but I'm having technical difficulties generating detailed coaching feedback. Please try again or check your Ollama setup.",
                'tokens_used': 0
            }
    
    def _call_ollama_fallback(self, prompt: str) -> Dict[str, Any]:
        """Fallback Ollama call with simplified prompt"""
        try:
            # Create a much simpler prompt
            simplified_prompt = self._simplify_prompt(prompt)
            
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": simplified_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,  # Lower temperature for more stable output
                    "num_predict": 500    # Shorter response
                }
            }
            
            logger.info("Attempting Ollama call with simplified prompt")
            
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
            
        except Exception as e:
            logger.error(f"Ollama fallback also failed: {e}")
            # Return a basic response
            return {
                'content': "I apologize, but I'm having technical difficulties generating detailed coaching feedback. Please try again or check your Ollama setup.",
                'tokens_used': 0
            }
    
    def _simplify_prompt(self, prompt: str) -> str:
        """Simplify the prompt to avoid Ollama issues"""
        return f"""
You are an expert interview coach. Provide brief, actionable feedback on this interview response.

Question: {prompt[:500] if len(prompt) > 500 else prompt}

Please provide:
1. One key strength
2. One area for improvement  
3. One specific action step

Keep your response under 200 words.
"""
    
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
    
    def _validate_ollama_model(self) -> None:
        """Validate that the configured Ollama model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                raise Exception(f"Ollama API not accessible: {response.status_code}")
            
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            logger.info(f"Available Ollama models: {available_models}")
            
            # Check for exact match
            if self.model_name in available_models:
                logger.info(f"Model {self.model_name} found")
                return
            
            # Check for models with tags (e.g., mistral:latest)
            for model in available_models:
                if model.startswith(self.model_name.split(':')[0]):
                    logger.info(f"Found similar model: {model}, using it instead")
                    self.model_name = model
                    return
            
            # Try fallback models
            fallback_models = ['llama2:7b', 'llama2:latest', 'mistral:latest', 'codellama:7b']
            for fallback in fallback_models:
                if fallback in available_models:
                    logger.info(f"Using fallback model: {fallback}")
                    self.model_name = fallback
                    return
            
            # If no fallbacks work, use the first available model
            if available_models:
                logger.warning(f"Configured model {self.model_name} not found, using {available_models[0]}")
                self.model_name = available_models[0]
                return
            
            raise Exception("No Ollama models available")
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise Exception(f"Ollama model validation failed: {e}")

    def _validate_ollama_model(self) -> None:
        """Validate that the configured Ollama model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                raise Exception(f"Ollama API not accessible: {response.status_code}")
            
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            logger.info(f"Available Ollama models: {available_models}")
            
            # Check for exact match
            if self.model_name in available_models:
                logger.info(f"Model {self.model_name} found")
                return
            
            # Check for models with tags (e.g., mistral:latest)
            for model in available_models:
                if model.startswith(self.model_name.split(':')[0]):
                    logger.info(f"Found similar model: {model}, using it instead")
                    self.model_name = model
                    return
            
            # Try fallback models
            fallback_models = ['llama2:7b', 'llama2:latest', 'mistral:latest', 'codellama:7b']
            for fallback in fallback_models:
                if fallback in available_models:
                    logger.info(f"Using fallback model: {fallback}")
                    self.model_name = fallback
                    return
            
            # If no fallbacks work, use the first available model
            if available_models:
                logger.warning(f"Configured model {self.model_name} not found, using {available_models[0]}")
                self.model_name = available_models[0]
                return
            
            raise Exception("No Ollama models available")
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise Exception(f"Ollama model validation failed: {e}")

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
