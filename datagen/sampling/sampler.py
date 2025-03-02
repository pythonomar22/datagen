"""
Text generation sampler module that handles different backend LLM APIs
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union
import json
import time

from datagen.config import SamplingConfig

logger = logging.getLogger(__name__)


class Sampler:
    """
    Text generation sampler that handles different LLM backend APIs
    """
    
    def __init__(self, config: SamplingConfig):
        """
        Initialize the sampler with configuration
        
        Args:
            config: Sampling configuration
        """
        self.config = config
        self._backend = None
        self._api_keys = {}
        
    def set_api_key(self, provider: str, api_key: str):
        """
        Set API key for a specific provider
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            api_key: The API key
        """
        self._api_keys[provider] = api_key
        
    def _get_api_key(self, provider: str) -> str:
        """Get API key for a provider, checking environment variables if not set"""
        if provider in self._api_keys:
            return self._api_keys[provider]
            
        # Try to get from environment variables
        env_var = f"{provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(f"API key for {provider} not found. Please set it via set_api_key() or as environment variable {env_var}")
            
        return api_key
        
    def sample(
        self, 
        prompt: str, 
        backend: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text based on a prompt
        
        Args:
            prompt: The input prompt
            backend: The backend to use (defaults to 'openai')
            model: The specific model to use
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated text
        """
        # Use provided values or defaults
        backend = backend or 'openai'
        
        # Override config values with kwargs
        params = {
            'temperature': kwargs.get('temperature', self.config.temperature),
            'top_p': kwargs.get('top_p', self.config.top_p),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            'frequency_penalty': kwargs.get('frequency_penalty', self.config.frequency_penalty),
            'presence_penalty': kwargs.get('presence_penalty', self.config.presence_penalty),
        }
        
        if self.config.top_k is not None:
            params['top_k'] = kwargs.get('top_k', self.config.top_k)
            
        # Select appropriate backend method
        if backend == 'openai':
            return self._sample_openai(prompt, model or 'gpt-3.5-turbo', **params)
        elif backend == 'anthropic':
            return self._sample_anthropic(prompt, model or 'claude-3-sonnet-20240229', **params)
        elif backend == 'huggingface':
            return self._sample_huggingface(prompt, model or 'meta-llama/Llama-2-7b-chat-hf', **params)
        elif backend == 'local':
            return self._sample_local(prompt, model, **params)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
            
    def _sample_openai(self, prompt: str, model: str, **params) -> str:
        """Sample text using OpenAI API"""
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package not installed. Install it with 'pip install openai'")
            
        openai.api_key = self._get_api_key('openai')
        
        # For newer API versions
        client = openai.OpenAI(api_key=openai.api_key)
        
        logger.debug(f"Generating with OpenAI model {model}: {prompt[:50]}...")
        start_time = time.time()
        
        # Adapt parameters for API
        api_params = {
            'model': model,
            'temperature': params.get('temperature'),
            'max_tokens': params.get('max_tokens'),
            'top_p': params.get('top_p'),
            'frequency_penalty': params.get('frequency_penalty'),
            'presence_penalty': params.get('presence_penalty'),
        }
        
        # Handle different model types
        if 'gpt-3.5' in model or 'gpt-4' in model:
            # Chat completion format
            response = client.chat.completions.create(
                messages=[{'role': 'user', 'content': prompt}],
                **api_params
            )
            result = response.choices[0].message.content
        else:
            # Legacy completion format
            response = client.completions.create(
                prompt=prompt,
                **api_params
            )
            result = response.choices[0].text
            
        elapsed_time = time.time() - start_time
        logger.debug(f"OpenAI response received in {elapsed_time:.2f}s")
        
        return result
        
    def _sample_anthropic(self, prompt: str, model: str, **params) -> str:
        """Sample text using Anthropic API"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Install it with 'pip install anthropic'")
            
        # Initialize client
        client = anthropic.Anthropic(api_key=self._get_api_key('anthropic'))
        
        logger.debug(f"Generating with Anthropic model {model}: {prompt[:50]}...")
        start_time = time.time()
        
        # Adapt parameters for API
        response = client.messages.create(
            model=model,
            max_tokens=params.get('max_tokens'),
            temperature=params.get('temperature'),
            top_p=params.get('top_p'),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.content[0].text
            
        elapsed_time = time.time() - start_time
        logger.debug(f"Anthropic response received in {elapsed_time:.2f}s")
        
        return result
    
    def _sample_huggingface(self, prompt: str, model: str, **params) -> str:
        """Sample text using Hugging Face Inference API"""
        try:
            import requests
        except ImportError:
            raise ImportError("Requests package not installed. Install it with 'pip install requests'")
            
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self._get_api_key('huggingface')}"}
        
        logger.debug(f"Generating with HuggingFace model {model}: {prompt[:50]}...")
        start_time = time.time()
        
        # Prepare payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": params.get('temperature'),
                "top_p": params.get('top_p'),
                "max_new_tokens": params.get('max_tokens'),
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Error from HuggingFace API: {response.text}")
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            
        result = response.json()[0].get('generated_text', '')
        if result.startswith(prompt):
            # Remove the prompt from the response if it's included
            result = result[len(prompt):]
            
        elapsed_time = time.time() - start_time
        logger.debug(f"HuggingFace response received in {elapsed_time:.2f}s")
        
        return result.strip()
        
    def _sample_local(self, prompt: str, model_path: str, **params) -> str:
        """Sample text using a local model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError:
            raise ImportError("Transformers package not installed. Install it with 'pip install transformers torch'")
            
        logger.debug(f"Generating with local model {model_path}: {prompt[:50]}...")
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"  # Use GPU if available
        )
        
        # Adapt parameters for API
        generation_params = {
            "max_new_tokens": params.get('max_tokens'),
            "temperature": params.get('temperature'),
            "top_p": params.get('top_p'),
            "do_sample": True,
        }
        
        if 'top_k' in params:
            generation_params['top_k'] = params.get('top_k')
            
        outputs = generator(prompt, **generation_params)
        result = outputs[0]["generated_text"]
        
        # Remove the prompt if it's included in the output
        if result.startswith(prompt):
            result = result[len(prompt):]
            
        elapsed_time = time.time() - start_time
        logger.debug(f"Local model response received in {elapsed_time:.2f}s")
        
        return result.strip()
    
    def batch_sample(
        self, 
        prompts: List[str], 
        backend: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: int = 10,
        **kwargs
    ) -> List[str]:
        """
        Generate text for a batch of prompts
        
        Args:
            prompts: List of input prompts
            backend: The backend to use
            model: The specific model to use
            batch_size: Number of prompts to process at once
            **kwargs: Additional sampling parameters
            
        Returns:
            List of generated texts
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(prompts) - 1)//batch_size + 1} ({len(batch)} prompts)")
            
            batch_results = []
            for prompt in batch:
                try:
                    result = self.sample(prompt, backend, model, **kwargs)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error generating text for prompt: {e}")
                    # Add an empty string as a placeholder for failed generations
                    batch_results.append("")
                    
            results.extend(batch_results)
            
        return results 