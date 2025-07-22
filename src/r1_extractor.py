#!/usr/bin/env python3
"""
Simple DeepSeek R1 reasoning token extractor.

This script provides a minimal interface to:
1. Query DeepSeek R1 API
2. Extract reasoning tokens from <think> tags
3. Save results to JSON

No complex dependencies - just requests and standard library.
"""
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests


class R1Extractor:
    """Simple client for extracting reasoning tokens from DeepSeek R1."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize R1 extractor.
        
        Args:
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            base_url: API base URL (defaults to DeepSeek's API)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY env var or pass api_key")
        
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Simple rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests
    
    def extract_reasoning_tokens(self, text: str) -> List[str]:
        """
        Extract reasoning tokens from <think> tags.
        
        Args:
            text: Response text containing <think> tags
            
        Returns:
            List of reasoning tokens
        """
        # Find all content within <think> tags
        think_pattern = r'<think>(.*?)</think>'
        matches = re.findall(think_pattern, text, re.DOTALL)
        
        reasoning_tokens = []
        for match in matches:
            # Split by newlines and clean up
            tokens = [
                token.strip() 
                for token in match.strip().split('\n') 
                if token.strip()
            ]
            reasoning_tokens.extend(tokens)
        
        return reasoning_tokens
    
    def query(
        self, 
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        include_thinking: bool = True
    ) -> Dict[str, any]:
        """
        Query DeepSeek R1 and extract reasoning.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            include_thinking: Whether to request thinking tokens
            
        Returns:
            Dictionary with response and extracted reasoning
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        # Prepare request
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        # Make request
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
            
            # Extract content and reasoning
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            reasoning_content = message.get("reasoning_content", "")
            
            # Extract reasoning tokens - check both content and reasoning_content
            reasoning_tokens = []
            
            # First try to extract from <think> tags in content
            if content:
                reasoning_tokens = self.extract_reasoning_tokens(content)
            
            # If no think tags found, use reasoning_content
            if not reasoning_tokens and reasoning_content:
                # Split reasoning content by newlines
                reasoning_tokens = [
                    token.strip() 
                    for token in reasoning_content.strip().split('\n') 
                    if token.strip()
                ]
            
            # Extract final answer
            if "</think>" in content:
                final_answer = content.split("</think>")[-1].strip()
            else:
                # If no think tags, the whole content is the answer
                final_answer = content.strip() if content else ""
                
                # If still no final answer and we have reasoning tokens, 
                # check if the last token contains the answer
                if not final_answer and reasoning_tokens:
                    last_token = reasoning_tokens[-1].lower()
                    if any(word in last_token for word in ['yes', 'no', 'true', 'false']):
                        final_answer = reasoning_tokens[-1]
            
            return {
                "prompt": prompt,
                "full_response": content,
                "reasoning_tokens": reasoning_tokens,
                "final_answer": final_answer,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": data.get("model"),
                    "usage": data.get("usage", {}),
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            }
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {
                "prompt": prompt,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def save_result(self, result: Dict, filepath: str):
        """Save extraction result to JSON file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved result to {filepath}")
    
    def load_result(self, filepath: str) -> Dict:
        """Load extraction result from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    """Example usage of R1Extractor."""
    # Example prompts
    example_prompts = [
        "What is 15 + 27?",
        "Is the Earth flat? Explain your reasoning.",
        "What is the capital of France and why?",
        "Solve: If a train travels 120 miles in 2 hours, what is its average speed?"
    ]
    
    # Initialize extractor
    try:
        extractor = R1Extractor()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your DeepSeek API key:")
        print("export DEEPSEEK_API_KEY='your-api-key-here'")
        return
    
    # Process each prompt
    results = []
    for i, prompt in enumerate(example_prompts):
        print(f"\n[{i+1}/{len(example_prompts)}] Processing: {prompt[:50]}...")
        
        result = extractor.query(prompt)
        
        if "error" not in result:
            print(f"  - Extracted {len(result['reasoning_tokens'])} reasoning tokens")
            print(f"  - Final answer: {result['final_answer'][:100]}...")
            
            # Save individual result
            extractor.save_result(result, f"outputs/r1_example_{i+1}.json")
            results.append(result)
        else:
            print(f"  - Error: {result['error']}")
    
    # Save all results
    if results:
        all_results = {
            "extraction_date": datetime.now().isoformat(),
            "num_prompts": len(example_prompts),
            "num_successful": len(results),
            "results": results
        }
        extractor.save_result(all_results, "outputs/r1_extraction_batch.json")
        
        # Print summary
        print(f"\n=== Extraction Summary ===")
        print(f"Processed: {len(example_prompts)} prompts")
        print(f"Successful: {len(results)}")
        total_tokens = sum(len(r['reasoning_tokens']) for r in results)
        print(f"Total reasoning tokens: {total_tokens}")


if __name__ == "__main__":
    main()