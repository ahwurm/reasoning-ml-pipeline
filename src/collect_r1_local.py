#!/usr/bin/env python3
"""
Batch data collection script for local DeepSeek R1 model.

Features:
- Load prompts from file or command line
- Batch processing for efficiency
- Progress tracking and resumable collection
- GPU memory management
- Save results incrementally
"""
import argparse
import json
import os
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from r1_local_extractor import R1LocalExtractor


class R1LocalDataCollector:
    """Batch data collector for local R1 model."""
    
    def __init__(
        self,
        output_dir: str = "data/r1_local_responses",
        checkpoint_file: str = "data/local_collection_checkpoint.json",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize local data collector.
        
        Args:
            output_dir: Directory to save responses
            checkpoint_file: File to track collection progress
            model_name: Model to use
            device: Device for inference
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_file = Path(checkpoint_file)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractor
        self.extractor = R1LocalExtractor(
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        
        # Load checkpoint if exists
        self.checkpoint = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict:
        """Load checkpoint data."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            "processed_prompts": [],
            "failed_prompts": [],
            "last_index": -1,
            "start_time": datetime.now().isoformat()
        }
    
    def _save_checkpoint(self):
        """Save checkpoint data."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def load_prompts(self, source: str) -> List[str]:
        """
        Load prompts from file or parse as direct input.
        
        Args:
            source: File path or comma-separated prompts
            
        Returns:
            List of prompts
        """
        # Check if it's a file
        if os.path.exists(source):
            with open(source, 'r') as f:
                # Support different formats
                content = f.read().strip()
                
                # Try JSON first
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and "prompts" in data:
                        return data["prompts"]
                except json.JSONDecodeError:
                    pass
                
                # Otherwise treat as line-separated
                return [line.strip() for line in content.split('\n') if line.strip()]
        
        # Otherwise treat as comma-separated prompts
        return [p.strip() for p in source.split(',') if p.strip()]
    
    def collect_batch(
        self,
        prompts: List[Tuple[int, str]],
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ) -> List[Dict[str, any]]:
        """
        Process a batch of prompts.
        
        Args:
            prompts: List of (index, prompt) tuples
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of results
        """
        results = []
        
        for idx, prompt in prompts:
            prompt_id = f"prompt_{idx:04d}"
            print(f"  Processing {prompt_id}: {prompt[:50]}...")
            
            try:
                # Generate response
                result = self.extractor.query(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stream=False  # No streaming for batch
                )
                
                if "error" not in result:
                    result["prompt_id"] = prompt_id
                    results.append((idx, result))
                    print(f"    ✓ Generated {len(result['reasoning_tokens'])} tokens")
                else:
                    print(f"    ✗ Error: {result['error']}")
                    results.append((idx, None))
                    
            except Exception as e:
                print(f"    ✗ Exception: {e}")
                results.append((idx, None))
        
        return results
    
    def collect(
        self,
        prompts: List[str],
        batch_size: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        resume: bool = True,
        stream_single: bool = True,
    ) -> Dict[str, any]:
        """
        Collect responses for all prompts.
        
        Args:
            prompts: List of prompts to process
            batch_size: Number of prompts to process together
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            resume: Whether to resume from checkpoint
            stream_single: Stream output when batch_size=1
            
        Returns:
            Collection statistics
        """
        # Load model first
        print("Loading model...")
        self.extractor.load_model()
        print()
        
        # Determine starting point
        start_idx = 0
        if resume and self.checkpoint["last_index"] >= 0:
            start_idx = self.checkpoint["last_index"] + 1
            print(f"Resuming from prompt {start_idx}")
        
        # Statistics
        stats = {
            "total_prompts": len(prompts),
            "successful": 0,
            "failed": 0,
            "skipped": start_idx,
            "start_time": datetime.now().isoformat(),
            "batch_size": batch_size,
            "device": self.extractor.device,
        }
        
        # Process prompts in batches
        for batch_start in range(start_idx, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = []
            
            # Prepare batch
            for i in range(batch_start, batch_end):
                prompt_id = f"prompt_{i:04d}"
                
                # Skip if already processed
                if prompt_id in self.checkpoint["processed_prompts"]:
                    print(f"Skipping {prompt_id} (already processed)")
                    stats["skipped"] += 1
                    continue
                
                batch_prompts.append((i, prompts[i]))
            
            if not batch_prompts:
                continue
            
            # Process batch
            print(f"\nProcessing batch {batch_start//batch_size + 1} "
                  f"({len(batch_prompts)} prompts)...")
            
            # Use streaming for single prompts
            if len(batch_prompts) == 1 and stream_single:
                idx, prompt = batch_prompts[0]
                prompt_id = f"prompt_{idx:04d}"
                
                print(f"[{idx+1}/{len(prompts)}] {prompt[:50]}...")
                result = self.extractor.query(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stream=True
                )
                
                if "error" not in result:
                    batch_results = [(idx, result)]
                else:
                    batch_results = [(idx, None)]
            else:
                batch_results = self.collect_batch(
                    batch_prompts,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens
                )
            
            # Save results
            for idx, result in batch_results:
                prompt_id = f"prompt_{idx:04d}"
                
                if result:
                    # Save result
                    result["prompt_id"] = prompt_id
                    result["collection_metadata"] = {
                        "batch_size": batch_size,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    output_file = self.output_dir / f"{prompt_id}.json"
                    self.extractor.save_result(result, str(output_file))
                    
                    # Update checkpoint
                    self.checkpoint["processed_prompts"].append(prompt_id)
                    self.checkpoint["last_index"] = idx
                    self._save_checkpoint()
                    
                    stats["successful"] += 1
                else:
                    # Record failure
                    self.checkpoint["failed_prompts"].append({
                        "prompt_id": prompt_id,
                        "prompt": prompts[idx],
                        "timestamp": datetime.now().isoformat()
                    })
                    self.checkpoint["last_index"] = idx
                    self._save_checkpoint()
                    stats["failed"] += 1
            
            # Clear GPU cache between batches if using CUDA
            if self.extractor.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final statistics
        stats["end_time"] = datetime.now().isoformat()
        stats["duration_seconds"] = (
            datetime.fromisoformat(stats["end_time"]) - 
            datetime.fromisoformat(stats["start_time"])
        ).total_seconds()
        
        # Add GPU memory stats if available
        if self.extractor.device == "cuda" and torch.cuda.is_available():
            stats["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        
        return stats
    
    def generate_summary(self) -> Dict[str, any]:
        """Generate summary of collected data."""
        summary = {
            "collection_info": {
                "output_dir": str(self.output_dir),
                "checkpoint_file": str(self.checkpoint_file),
                "model": self.extractor.model_name,
                "device": self.extractor.device,
                "quantization": {
                    "8bit": self.extractor.load_in_8bit,
                    "4bit": self.extractor.load_in_4bit
                },
                "generated_at": datetime.now().isoformat()
            },
            "statistics": {
                "total_processed": len(self.checkpoint["processed_prompts"]),
                "total_failed": len(self.checkpoint["failed_prompts"]),
            },
            "responses": []
        }
        
        # Analyze each response
        total_tokens = 0
        total_time = 0
        
        for prompt_id in self.checkpoint["processed_prompts"]:
            filepath = self.output_dir / f"{prompt_id}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    num_tokens = len(data.get("reasoning_tokens", []))
                    gen_time = data["metadata"].get("generation_time", 0)
                    
                    total_tokens += num_tokens
                    total_time += gen_time
                    
                    summary["responses"].append({
                        "prompt_id": prompt_id,
                        "prompt_preview": data["prompt"][:100] + "...",
                        "num_reasoning_tokens": num_tokens,
                        "generation_time": gen_time,
                        "tokens_per_second": num_tokens / gen_time if gen_time > 0 else 0,
                        "timestamp": data["metadata"]["timestamp"]
                    })
        
        # Add aggregated stats
        if summary["responses"]:
            summary["statistics"]["avg_reasoning_tokens"] = total_tokens / len(summary["responses"])
            summary["statistics"]["avg_generation_time"] = total_time / len(summary["responses"])
            summary["statistics"]["total_reasoning_tokens"] = total_tokens
            summary["statistics"]["total_generation_time"] = total_time
            
            # Calculate tokens per second
            if total_time > 0:
                summary["statistics"]["overall_tokens_per_second"] = total_tokens / total_time
        
        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect reasoning data from local DeepSeek R1 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect from prompts file
  python collect_r1_local.py prompts.txt
  
  # Collect with 8-bit quantization
  python collect_r1_local.py prompts.txt --8bit
  
  # Process in batches (faster but uses more memory)
  python collect_r1_local.py prompts.txt --batch-size 4
  
  # Use specific model
  python collect_r1_local.py prompts.txt --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  
  # Generate summary
  python collect_r1_local.py --summary
        """
    )
    
    parser.add_argument(
        "prompts",
        nargs="?",
        help="Prompts file or comma-separated prompts"
    )
    parser.add_argument(
        "--output-dir",
        default="data/r1_local_responses",
        help="Output directory for responses"
    )
    parser.add_argument(
        "--checkpoint-file",
        default="data/local_collection_checkpoint.json",
        help="Checkpoint file for resuming"
    )
    parser.add_argument(
        "--model",
        help="Model name (default: DeepSeek-R1-Distill-Qwen-7B)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore checkpoint"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate summary of collected data"
    )
    
    args = parser.parse_args()
    
    # Check for required libraries
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("Error: Required dependencies not installed.")
        print("Install with: pip install -r requirements-local.txt")
        sys.exit(1)
    
    # Initialize collector
    collector = R1LocalDataCollector(
        output_dir=args.output_dir,
        checkpoint_file=args.checkpoint_file,
        model_name=args.model,
        device=args.device,
        load_in_8bit=getattr(args, '8bit'),
        load_in_4bit=getattr(args, '4bit'),
    )
    
    # Generate summary if requested
    if args.summary:
        summary = collector.generate_summary()
        print(json.dumps(summary, indent=2))
        
        # Save summary
        summary_file = Path(args.output_dir) / "local_collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_file}")
        return
    
    # Check for prompts
    if not args.prompts:
        print("Error: No prompts provided")
        parser.print_help()
        sys.exit(1)
    
    # Load prompts
    try:
        prompts = collector.load_prompts(args.prompts)
        print(f"Loaded {len(prompts)} prompts")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        sys.exit(1)
    
    # Collect data
    print(f"\nStarting local data collection...")
    print(f"Model: {collector.extractor.model_name}")
    print(f"Device: {collector.extractor.device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Resume: {not args.no_resume}")
    
    if getattr(args, '8bit'):
        print("Quantization: 8-bit")
    elif getattr(args, '4bit'):
        print("Quantization: 4-bit")
    
    print()
    
    stats = collector.collect(
        prompts,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        resume=not args.no_resume,
        stream_single=not args.no_stream
    )
    
    # Print final statistics
    print(f"\n=== Collection Complete ===")
    print(f"Total prompts: {stats['total_prompts']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Duration: {stats['duration_seconds']:.1f} seconds")
    
    if stats.get("gpu_memory_used_gb"):
        print(f"GPU memory used: {stats['gpu_memory_used_gb']:.2f} GB")
    
    # Generate and save summary
    summary = collector.generate_summary()
    summary_file = Path(args.output_dir) / "local_collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    
    if summary["statistics"].get("overall_tokens_per_second"):
        print(f"Average speed: {summary['statistics']['overall_tokens_per_second']:.1f} tokens/second")


if __name__ == "__main__":
    main()