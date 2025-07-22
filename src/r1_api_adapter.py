#!/usr/bin/env python3
"""
Adapter for DeepSeek API that handles the new reasoning format.
"""
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional
import requests


class R1APIAdapter:
    """Adapter for DeepSeek's reasoning API format."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize adapter with API key."""
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")
        
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Reduced for better performance
    
    def query_binary_decision(self, prompt: str, max_tokens: int = 1000) -> Dict:
        """
        Query API for a binary decision with reasoning.
        
        Optimized for Yes/No questions with step-by-step reasoning.
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        # Ensure prompt asks for binary answer
        if not any(phrase in prompt.lower() for phrase in ['yes or no', 'yes/no', 'true or false']):
            prompt += "\nAnswer with Yes or No."
        
        # API request
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,  # Lower for more consistent answers
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            self.last_request_time = time.time()
            
            # Parse response
            data = response.json()
            choice = data['choices'][0]
            message = choice['message']
            
            # Extract reasoning and answer
            reasoning_content = message.get('reasoning_content', '')
            content = message.get('content', '')
            
            # Parse reasoning into tokens
            reasoning_tokens = []
            if reasoning_content:
                # Split by sentences or newlines
                sentences = re.split(r'[.!?]\s+|\n+', reasoning_content)
                reasoning_tokens = [s.strip() for s in sentences if s.strip()]
            
            # Extract binary answer
            answer = self._extract_binary_answer(reasoning_content, content, reasoning_tokens)
            
            return {
                "success": True,
                "prompt": prompt,
                "reasoning_tokens": reasoning_tokens,
                "reasoning_full": reasoning_content,
                "content": content,
                "answer": answer,
                "metadata": {
                    "model": data.get("model"),
                    "usage": data.get("usage", {}),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }
    
    def _extract_binary_answer(self, reasoning: str, content: str, tokens: List[str]) -> str:
        """Extract Yes/No answer from various sources."""
        # Check content first (most direct)
        if content:
            content_lower = content.lower().strip()
            if content_lower in ['yes', 'no']:
                return content_lower.capitalize()
            if 'yes' in content_lower and 'no' not in content_lower:
                return "Yes"
            if 'no' in content_lower and 'yes' not in content_lower:
                return "No"
        
        # Check last reasoning tokens
        for token in reversed(tokens[-3:]):  # Check last 3 tokens
            token_lower = token.lower()
            # Look for definitive statements
            if re.search(r'\b(the answer is|therefore|thus|so)\s+(yes|no)\b', token_lower):
                match = re.search(r'\b(yes|no)\b', token_lower)
                if match:
                    return match.group().capitalize()
            # Direct yes/no at end of sentence
            if re.search(r'\b(yes|no)\s*[.,!?]?\s*$', token_lower):
                match = re.search(r'\b(yes|no)\b', token_lower)
                if match:
                    return match.group().capitalize()
        
        # Check full reasoning as last resort
        if reasoning:
            reasoning_lower = reasoning.lower()
            # Look for clear answer patterns
            patterns = [
                r'the answer is\s+(yes|no)',
                r'therefore[,:]?\s+(yes|no)',
                r'so[,:]?\s+(yes|no)',
                r'thus[,:]?\s+(yes|no)',
                r'my answer is\s+(yes|no)',
                r'final answer[:]?\s+(yes|no)'
            ]
            for pattern in patterns:
                match = re.search(pattern, reasoning_lower)
                if match:
                    return match.group(1).capitalize()
        
        return "Unknown"