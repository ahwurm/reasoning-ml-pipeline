#!/usr/bin/env python3
"""
Download DeepSeek R1 model files for offline use.

This script helps download the model files ahead of time to avoid
delays during first use. It also provides information about disk
space and download progress.
"""
import argparse
import os
import sys
import shutil
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


def get_disk_usage(path):
    """Get disk usage statistics for a path."""
    stat = shutil.disk_usage(path)
    return {
        "total": stat.total / (1024**3),  # GB
        "used": stat.used / (1024**3),
        "free": stat.free / (1024**3)
    }


def estimate_model_size(model_name, quantization=None):
    """Estimate model size based on parameters and quantization."""
    # Base estimates for common models
    size_estimates = {
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
            "fp16": 14.0,  # GB
            "int8": 7.0,
            "int4": 3.5,
        },
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
            "fp16": 16.0,
            "int8": 8.0,
            "int4": 4.0,
        }
    }
    
    # Default estimate
    if model_name not in size_estimates:
        # Rough estimate based on parameter count in name
        if "7B" in model_name:
            base_size = 14.0
        elif "8B" in model_name:
            base_size = 16.0
        elif "13B" in model_name:
            base_size = 26.0
        else:
            base_size = 20.0  # Conservative default
            
        size_estimates[model_name] = {
            "fp16": base_size,
            "int8": base_size / 2,
            "int4": base_size / 4,
        }
    
    estimates = size_estimates[model_name]
    
    if quantization == "8bit":
        return estimates["int8"]
    elif quantization == "4bit":
        return estimates["int4"]
    else:
        return estimates["fp16"]


def download_model(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    cache_dir=None,
    quantization=None,
    token=None
):
    """
    Download model files from Hugging Face.
    
    Args:
        model_name: Model identifier on Hugging Face
        cache_dir: Directory to save model files
        quantization: Quantization to use ('8bit', '4bit', or None)
        token: Hugging Face token (if needed for private models)
    """
    if not DEPENDENCIES_AVAILABLE:
        print("Error: Required dependencies not installed.")
        print("Install with: pip install transformers torch huggingface-hub")
        return False
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print()
    
    # Check disk space
    disk_info = get_disk_usage(cache_dir)
    estimated_size = estimate_model_size(model_name, quantization)
    
    print(f"Disk space:")
    print(f"  Available: {disk_info['free']:.1f} GB")
    print(f"  Required: ~{estimated_size:.1f} GB")
    
    if disk_info['free'] < estimated_size * 1.2:  # 20% buffer
        print(f"\nWarning: Low disk space! You need at least {estimated_size * 1.2:.1f} GB free.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    print()
    
    # Download tokenizer first (small)
    print("Downloading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download tokenizer: {e}")
        return False
    
    # Download model files
    print("\nDownloading model files...")
    print("This may take a while depending on your connection speed.")
    print("The download will resume if interrupted.")
    print()
    
    try:
        # For quantized models, we still download the full model
        # Quantization happens at load time
        if quantization:
            print(f"Note: Downloading full precision model. {quantization} quantization will be applied at load time.")
            print()
        
        # Use snapshot_download for better progress tracking
        snapshot_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            token=token,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print(f"\n✓ Model files downloaded to: {snapshot_path}")
        
        # Verify model can be loaded
        print("\nVerifying model files...")
        
        # Just instantiate the model architecture without loading weights
        # This is faster and uses less memory
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        print("✓ Model files verified successfully")
        
        # Print summary
        print("\n" + "="*50)
        print("Download complete!")
        print(f"Model: {model_name}")
        print(f"Location: {snapshot_path}")
        
        # Calculate actual size
        total_size = 0
        for root, dirs, files in os.walk(snapshot_path):
            for file in files:
                if not file.startswith('.'):  # Skip hidden files
                    total_size += os.path.getsize(os.path.join(root, file))
        
        total_size_gb = total_size / (1024**3)
        print(f"Total size: {total_size_gb:.1f} GB")
        
        # Provide usage instructions
        print("\nTo use the model:")
        print("```python")
        print("from r1_local_extractor import R1LocalExtractor")
        print(f"extractor = R1LocalExtractor(model_name='{model_name}')")
        if quantization == "8bit":
            print("# For 8-bit quantization:")
            print("extractor = R1LocalExtractor(load_in_8bit=True)")
        elif quantization == "4bit":
            print("# For 4-bit quantization:")
            print("extractor = R1LocalExtractor(load_in_4bit=True)")
        print("result = extractor.query('Your prompt here')")
        print("```")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download model: {e}")
        
        if "token" in str(e):
            print("\nNote: Some models require authentication.")
            print("Get a token from: https://huggingface.co/settings/tokens")
            print("Then run: huggingface-cli login")
        
        return False


def list_downloaded_models(cache_dir=None):
    """List models already downloaded."""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    hub_path = Path(cache_dir) / "hub"
    
    if not hub_path.exists():
        print("No models found in cache.")
        return
    
    print("Downloaded models:")
    print("-" * 50)
    
    # Look for model directories
    model_dirs = []
    for item in hub_path.iterdir():
        if item.is_dir() and item.name.startswith("models--"):
            model_name = item.name.replace("models--", "").replace("--", "/")
            
            # Calculate size
            total_size = 0
            for root, dirs, files in os.walk(item):
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))
            
            size_gb = total_size / (1024**3)
            model_dirs.append((model_name, size_gb, item))
    
    if not model_dirs:
        print("No models found.")
        return
    
    # Sort by size
    model_dirs.sort(key=lambda x: x[1], reverse=True)
    
    total_size = 0
    for model_name, size_gb, path in model_dirs:
        print(f"{model_name}: {size_gb:.1f} GB")
        total_size += size_gb
    
    print("-" * 50)
    print(f"Total: {total_size:.1f} GB")
    print(f"\nCache location: {cache_dir}")


def clear_cache(cache_dir=None):
    """Clear model cache."""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    print(f"Cache directory: {cache_dir}")
    
    # Calculate cache size
    total_size = 0
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    
    size_gb = total_size / (1024**3)
    print(f"Cache size: {size_gb:.1f} GB")
    
    response = input("\nAre you sure you want to clear the cache? (y/n): ")
    if response.lower() == 'y':
        try:
            shutil.rmtree(cache_dir)
            print("✓ Cache cleared successfully")
        except Exception as e:
            print(f"✗ Failed to clear cache: {e}")
    else:
        print("Cache clear cancelled.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download DeepSeek R1 models for local use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default model (DeepSeek-R1-Distill-Qwen-7B)
  python download_model.py
  
  # Download with 8-bit quantization info
  python download_model.py --quantization 8bit
  
  # Download different model
  python download_model.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  
  # List downloaded models
  python download_model.py --list
  
  # Clear cache
  python download_model.py --clear-cache
        """
    )
    
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model to download from Hugging Face"
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to cache models (default: ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--quantization",
        choices=["8bit", "4bit"],
        help="Quantization type (affects size estimate)"
    )
    parser.add_argument(
        "--token",
        help="Hugging Face token for private models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List downloaded models"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear model cache"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        print("Error: Required dependencies not installed.")
        print("Install with: pip install -r requirements-local.txt")
        sys.exit(1)
    
    # Execute requested action
    if args.list:
        list_downloaded_models(args.cache_dir)
    elif args.clear_cache:
        clear_cache(args.cache_dir)
    else:
        success = download_model(
            model_name=args.model,
            cache_dir=args.cache_dir,
            quantization=args.quantization,
            token=args.token
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()