#!/usr/bin/env python3
"""
Batch data collection script for DeepSeek R1.

Features:
- Load prompts from file or command line
- Progress tracking and resumable collection
- Rate limiting and retry logic
- Save results incrementally
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from r1_extractor import R1Extractor


class R1DataCollector:
    """Batch data collector for R1 reasoning traces."""
    
    def __init__(
        self,
        output_dir: str = "data/r1_responses",
        checkpoint_file: str = "data/collection_checkpoint.json"
    ):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to save responses
            checkpoint_file: File to track collection progress
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_file = Path(checkpoint_file)
        self.extractor = R1Extractor()
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
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
    
    def collect(
        self,
        prompts: List[str],
        max_retries: int = 3,
        retry_delay: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        resume: bool = True
    ) -> Dict[str, any]:
        """
        Collect responses for all prompts.
        
        Args:
            prompts: List of prompts to process
            max_retries: Maximum retries per prompt
            retry_delay: Seconds to wait between retries
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            resume: Whether to resume from checkpoint
            
        Returns:
            Collection statistics
        """
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
            "start_time": datetime.now().isoformat()
        }
        
        # Process each prompt
        for i in range(start_idx, len(prompts)):
            prompt = prompts[i]
            prompt_id = f"prompt_{i:04d}"
            
            print(f"\n[{i+1}/{len(prompts)}] Processing prompt {prompt_id}")
            print(f"  Prompt: {prompt[:100]}...")
            
            # Skip if already processed
            if prompt_id in self.checkpoint["processed_prompts"]:
                print("  Already processed, skipping")
                stats["skipped"] += 1
                continue
            
            # Try to collect with retries
            success = False
            for attempt in range(max_retries):
                try:
                    # Query R1
                    result = self.extractor.query(
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    if "error" not in result:
                        # Save result
                        result["prompt_id"] = prompt_id
                        result["collection_metadata"] = {
                            "attempt": attempt + 1,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        output_file = self.output_dir / f"{prompt_id}.json"
                        self.extractor.save_result(result, str(output_file))
                        
                        # Update checkpoint
                        self.checkpoint["processed_prompts"].append(prompt_id)
                        self.checkpoint["last_index"] = i
                        self._save_checkpoint()
                        
                        # Update stats
                        stats["successful"] += 1
                        success = True
                        
                        print(f"  Success! Extracted {len(result['reasoning_tokens'])} tokens")
                        break
                    
                    else:
                        print(f"  Attempt {attempt+1} failed: {result['error']}")
                        
                except Exception as e:
                    print(f"  Attempt {attempt+1} error: {e}")
                
                # Retry delay
                if attempt < max_retries - 1:
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            
            if not success:
                # Record failure
                self.checkpoint["failed_prompts"].append({
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat()
                })
                self.checkpoint["last_index"] = i
                self._save_checkpoint()
                stats["failed"] += 1
                print(f"  Failed after {max_retries} attempts")
        
        # Final statistics
        stats["end_time"] = datetime.now().isoformat()
        stats["duration_seconds"] = (
            datetime.fromisoformat(stats["end_time"]) - 
            datetime.fromisoformat(stats["start_time"])
        ).total_seconds()
        
        return stats
    
    def generate_summary(self) -> Dict[str, any]:
        """Generate summary of collected data."""
        summary = {
            "collection_info": {
                "output_dir": str(self.output_dir),
                "checkpoint_file": str(self.checkpoint_file),
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
        total_final_length = 0
        
        for prompt_id in self.checkpoint["processed_prompts"]:
            filepath = self.output_dir / f"{prompt_id}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    num_tokens = len(data.get("reasoning_tokens", []))
                    final_length = len(data.get("final_answer", ""))
                    
                    total_tokens += num_tokens
                    total_final_length += final_length
                    
                    summary["responses"].append({
                        "prompt_id": prompt_id,
                        "prompt_preview": data["prompt"][:100] + "...",
                        "num_reasoning_tokens": num_tokens,
                        "final_answer_length": final_length,
                        "timestamp": data["metadata"]["timestamp"]
                    })
        
        # Add aggregated stats
        if summary["responses"]:
            summary["statistics"]["avg_reasoning_tokens"] = total_tokens / len(summary["responses"])
            summary["statistics"]["avg_answer_length"] = total_final_length / len(summary["responses"])
            summary["statistics"]["total_reasoning_tokens"] = total_tokens
        
        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect reasoning data from DeepSeek R1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect from prompts file
  python collect_r1_data.py prompts.txt
  
  # Collect from JSON file
  python collect_r1_data.py prompts.json
  
  # Direct prompts
  python collect_r1_data.py "What is 2+2?, Explain gravity"
  
  # Custom output directory
  python collect_r1_data.py prompts.txt --output-dir my_data
  
  # Don't resume from checkpoint
  python collect_r1_data.py prompts.txt --no-resume
        """
    )
    
    parser.add_argument(
        "prompts",
        help="Prompts file or comma-separated prompts"
    )
    parser.add_argument(
        "--output-dir",
        default="data/r1_responses",
        help="Output directory for responses"
    )
    parser.add_argument(
        "--checkpoint-file",
        default="data/collection_checkpoint.json",
        help="Checkpoint file for resuming"
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
        help="Maximum response tokens"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per prompt"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore checkpoint"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate summary of collected data"
    )
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = R1DataCollector(
        output_dir=args.output_dir,
        checkpoint_file=args.checkpoint_file
    )
    
    # Generate summary if requested
    if args.summary:
        summary = collector.generate_summary()
        print(json.dumps(summary, indent=2))
        
        # Save summary
        summary_file = Path(args.output_dir) / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_file}")
        return
    
    # Load prompts
    try:
        prompts = collector.load_prompts(args.prompts)
        print(f"Loaded {len(prompts)} prompts")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        sys.exit(1)
    
    # Collect data
    print(f"\nStarting data collection...")
    print(f"Output directory: {args.output_dir}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Resume: {not args.no_resume}")
    
    stats = collector.collect(
        prompts,
        max_retries=args.max_retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        resume=not args.no_resume
    )
    
    # Print final statistics
    print(f"\n=== Collection Complete ===")
    print(f"Total prompts: {stats['total_prompts']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Duration: {stats['duration_seconds']:.1f} seconds")
    
    # Generate and save summary
    summary = collector.generate_summary()
    summary_file = Path(args.output_dir) / "collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()