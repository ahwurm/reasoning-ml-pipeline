#!/usr/bin/env python3
"""
Local DeepSeek R1 reasoning token extractor using Hugging Face model.

This script provides a local alternative to the API-based extractor:
1. Load DeepSeek-R1-Distill-Qwen-7B from Hugging Face
2. Generate responses with reasoning tokens
3. Extract reasoning tokens from <think> tags
4. Save results to JSON

Requires transformers and torch.
"""
import json
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TextStreamer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/torch not installed. Install with: pip install transformers torch")


class R1LocalExtractor:
    """Local model client for extracting reasoning tokens from DeepSeek R1."""
    
    # Default model name
    DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize local R1 extractor.
        
        Args:
            model_name: Hugging Face model name (defaults to DeepSeek-R1-Distill-Qwen-7B)
            device: Device to use ('cuda', 'cpu', 'mps', or None for auto)
            load_in_8bit: Use 8-bit quantization (reduces memory)
            load_in_4bit: Use 4-bit quantization (reduces memory even more)
            cache_dir: Directory to cache model files
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        
        # Device selection
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Using device: {self.device}")
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Load model on first use
        self._loaded = False
    
    def load_model(self):
        """Load model and tokenizer."""
        if self._loaded:
            return
        
        print(f"Loading model: {self.model_name}")
        print("This may take a while on first run...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Quantization config
        quantization_config = None
        if self.load_in_4bit or self.load_in_8bit:
            if not torch.cuda.is_available():
                print("Warning: Quantization requires CUDA. Falling back to full precision.")
                self.load_in_4bit = self.load_in_8bit = False
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16 if self.load_in_4bit else None,
                )
        
        # Model loading arguments
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        elif self.device != "cpu":
            model_kwargs["device_map"] = "auto"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if self.device == "cpu" or (self.device == "mps" and not quantization_config):
            self.model = self.model.to(self.device)
        
        # Set model to eval mode
        self.model.eval()
        
        self._loaded = True
        print(f"Model loaded successfully!")
        
        # Print memory usage if on GPU
        if self.device == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU memory used: {memory_used:.2f} GB")
    
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
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        stream: bool = False,
    ) -> str:
        """
        Generate response using local model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            stream: Whether to stream output
            
        Returns:
            Generated text
        """
        # Ensure model is loaded
        self.load_model()
        
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add streamer if requested
        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            gen_kwargs["streamer"] = streamer
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        
        return generated
    
    def query(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        include_thinking: bool = True,
        stream: bool = False,
    ) -> Dict[str, any]:
        """
        Query local model and extract reasoning (compatible with API version).
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum response tokens
            temperature: Sampling temperature
            include_thinking: Whether to request thinking tokens
            stream: Whether to stream output
            
        Returns:
            Dictionary with response and extracted reasoning
        """
        start_time = time.time()
        
        # For R1 models, we might need to add a system prompt to encourage thinking
        if include_thinking and "<think>" not in prompt:
            # You can customize this based on what works best
            full_prompt = f"{prompt}\n\nPlease think step by step about this problem."
        else:
            full_prompt = prompt
        
        try:
            # Generate response
            response = self.generate(
                full_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stream=stream
            )
            
            # Extract reasoning tokens
            reasoning_tokens = self.extract_reasoning_tokens(response)
            
            # Extract final answer (everything after last </think>)
            final_answer = response
            if "</think>" in response:
                final_answer = response.split("</think>")[-1].strip()
            
            # Calculate tokens (approximate)
            input_tokens = len(self.tokenizer.encode(full_prompt))
            output_tokens = len(self.tokenizer.encode(response))
            
            generation_time = time.time() - start_time
            
            return {
                "prompt": prompt,
                "full_response": response,
                "reasoning_tokens": reasoning_tokens,
                "final_answer": final_answer,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_name,
                    "device": self.device,
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "generation_time": generation_time,
                    "load_in_8bit": self.load_in_8bit,
                    "load_in_4bit": self.load_in_4bit,
                }
            }
            
        except Exception as e:
            print(f"Generation failed: {e}")
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
    """Example usage of R1LocalExtractor."""
    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers and torch are required.")
        print("Install with: pip install transformers torch")
        return
    
    # Example prompts
    example_prompts = [
        "What is 15 + 27?",
        "Is the Earth flat? Explain your reasoning.",
        "What is the capital of France and why?",
    ]
    
    # Initialize extractor
    print("Initializing local R1 extractor...")
    print("Note: First run will download ~15GB model files")
    print()
    
    # Check for GPU and suggest options
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU detected with {gpu_memory:.1f}GB memory")
        
        if gpu_memory < 16:
            print("Tip: Use --8bit or --4bit flag for lower memory usage")
            use_8bit = input("Use 8-bit quantization? (y/n): ").lower() == 'y'
        else:
            use_8bit = False
    else:
        print("No GPU detected, using CPU (will be slower)")
        use_8bit = False
    
    # Create extractor
    extractor = R1LocalExtractor(load_in_8bit=use_8bit)
    
    # Process each prompt
    results = []
    for i, prompt in enumerate(example_prompts):
        print(f"\n[{i+1}/{len(example_prompts)}] Processing: {prompt[:50]}...")
        
        result = extractor.query(prompt, stream=True)
        
        if "error" not in result:
            print(f"\n  - Extracted {len(result['reasoning_tokens'])} reasoning tokens")
            print(f"  - Generation time: {result['metadata']['generation_time']:.2f}s")
            
            # Save individual result
            extractor.save_result(result, f"outputs/r1_local_example_{i+1}.json")
            results.append(result)
        else:
            print(f"  - Error: {result['error']}")
    
    # Save all results
    if results:
        all_results = {
            "extraction_date": datetime.now().isoformat(),
            "num_prompts": len(example_prompts),
            "num_successful": len(results),
            "model": extractor.model_name,
            "device": extractor.device,
            "results": results
        }
        extractor.save_result(all_results, "outputs/r1_local_extraction_batch.json")
        
        # Print summary
        print(f"\n=== Local Extraction Summary ===")
        print(f"Processed: {len(example_prompts)} prompts")
        print(f"Successful: {len(results)}")
        total_tokens = sum(len(r['reasoning_tokens']) for r in results)
        print(f"Total reasoning tokens: {total_tokens}")
        
        total_time = sum(r['metadata']['generation_time'] for r in results)
        print(f"Total generation time: {total_time:.2f}s")
        print(f"Average time per prompt: {total_time/len(results):.2f}s")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Local R1 reasoning extraction")
    parser.add_argument("--8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    # Override settings based on args
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    main()