#!/usr/bin/env python3
"""
Incremental dataset generation with resume capability.
Generates samples in small batches and appends to existing dataset.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/src')

from generate_binary_ddm_dataset import BinaryDDMDatasetGenerator


class IncrementalDatasetGenerator:
    """Generate dataset incrementally with resume capability."""
    
    def __init__(self, api_key: str, output_path: str):
        """Initialize incremental generator."""
        self.output_path = Path(output_path)
        self.api_key = api_key
        self.dataset = self._load_or_create_dataset()
        self.generator = BinaryDDMDatasetGenerator(
            use_local_model=False,
            api_key=api_key
        )
        
    def _load_or_create_dataset(self) -> Dict:
        """Load existing dataset or create new one."""
        if self.output_path.exists():
            print(f"Loading existing dataset from {self.output_path}")
            with open(self.output_path, 'r') as f:
                dataset = json.load(f)
            print(f"Found {len(dataset['samples'])} existing samples")
            return dataset
        else:
            print("Creating new dataset")
            return {
                "dataset_info": {
                    "version": "1.0",
                    "task_type": "binary_decision",
                    "categories": ["math", "comparison", "prime", "divisibility", "pattern"],
                    "difficulty_distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
                    "generation_method": "incremental_api",
                    "creation_date": datetime.now().isoformat()
                },
                "samples": []
            }
    
    def _save_dataset(self):
        """Save dataset to file."""
        # Create backup of existing file
        if self.output_path.exists():
            backup_path = self.output_path.with_suffix('.backup')
            self.output_path.rename(backup_path)
        
        # Save new version
        try:
            with open(self.output_path, 'w') as f:
                json.dump(self.dataset, f, indent=2)
            
            # Remove backup on success
            backup_path = self.output_path.with_suffix('.backup')
            if backup_path.exists():
                backup_path.unlink()
                
        except Exception as e:
            print(f"Error saving dataset: {e}")
            # Restore backup
            backup_path = self.output_path.with_suffix('.backup')
            if backup_path.exists():
                backup_path.rename(self.output_path)
            raise
    
    def _update_dataset_info(self):
        """Update dataset metadata."""
        samples = self.dataset["samples"]
        
        # Count answers
        yes_count = sum(1 for s in samples if s["correct_answer"] == "Yes")
        no_count = sum(1 for s in samples if s["correct_answer"] == "No")
        
        # Update info
        self.dataset["dataset_info"]["total_samples"] = len(samples)
        self.dataset["dataset_info"]["last_updated"] = datetime.now().isoformat()
        self.dataset["dataset_info"]["balance"] = {
            "yes": yes_count,
            "no": no_count,
            "ratio": yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
        }
    
    def generate_samples(self, num_samples: int, max_retries: int = 3):
        """Generate specified number of samples and append to dataset."""
        start_id = len(self.dataset["samples"])
        target_samples = start_id + num_samples
        
        print(f"\nGenerating {num_samples} new samples")
        print(f"Starting from sample ID: {start_id}")
        print(f"Target total: {target_samples}")
        
        categories = ["math", "comparison", "prime", "divisibility", "pattern"]
        difficulties = ["easy", "medium", "hard"]
        difficulty_weights = [0.3, 0.5, 0.2]
        
        generated = 0
        failed = 0
        
        while len(self.dataset["samples"]) < target_samples and failed < num_samples:
            sample_id = len(self.dataset["samples"])
            
            # Generate task
            category = categories[sample_id % len(categories)]  # Rotate through categories
            difficulty = np.random.choice(difficulties, p=difficulty_weights) if 'np' in globals() else difficulties[1]
            
            # Get generator method
            generator_map = {
                "math": self.generator.generate_math_verification_task,
                "comparison": self.generator.generate_comparison_task,
                "prime": self.generator.generate_prime_check_task,
                "divisibility": self.generator.generate_divisibility_task,
                "pattern": self.generator.generate_pattern_task
            }
            
            task = generator_map[category](difficulty)
            
            print(f"\n[{sample_id + 1}] {task['prompt'][:60]}...")
            
            # Generate reasoning with retries
            reasoning_result = None
            for retry in range(max_retries):
                try:
                    reasoning_result = self.generator.generate_reasoning_trace(task)
                    if reasoning_result["model_answer"] != "Unknown":
                        break
                    else:
                        print(f"  Retry {retry + 1}: Unknown answer")
                except Exception as e:
                    print(f"  Retry {retry + 1}: {str(e)[:50]}")
                    time.sleep(2)  # Brief pause before retry
            
            if not reasoning_result or reasoning_result["model_answer"] == "Unknown":
                print(f"  Failed after {max_retries} retries")
                failed += 1
                continue
            
            # Calculate trajectory
            trajectory = self.generator.calculate_evidence_trajectory(
                reasoning_result["reasoning_tokens"],
                task["correct_answer"]
            )
            
            # Create sample
            sample = {
                "id": f"binary_{sample_id:05d}",
                "prompt": task["prompt"],
                "category": task["category"],
                "subcategory": task.get("subcategory", ""),
                "difficulty": task["difficulty"],
                "correct_answer": task["correct_answer"],
                "model_answer": reasoning_result["model_answer"],
                "is_correct": reasoning_result["model_answer"] == task["correct_answer"],
                "reasoning_trace": {
                    "tokens": reasoning_result["reasoning_tokens"],
                    "evidence_scores": trajectory["evidence_scores"],
                    "timestamps": trajectory["timestamps"],
                    "generation_time": reasoning_result["generation_time"]
                },
                "ddm_trajectory": {
                    "cumulative_evidence": trajectory["cumulative_evidence"],
                    "decision_time": trajectory["decision_time"],
                    "reached_threshold": trajectory["reached_threshold"],
                    "final_evidence": trajectory["final_evidence"]
                },
                "metadata": task.get("metadata", {})
            }
            
            # Add to dataset
            self.dataset["samples"].append(sample)
            generated += 1
            
            print(f"  ✓ Answer: {reasoning_result['model_answer']} (Correct: {task['correct_answer']})")
            print(f"  Tokens: {len(reasoning_result['reasoning_tokens'])}")
            
            # Save after each sample
            self._update_dataset_info()
            self._save_dataset()
            
            # Progress update every 10 samples
            if generated % 10 == 0:
                print(f"\nProgress: {len(self.dataset['samples'])}/{target_samples} total samples")
                balance = self.dataset["dataset_info"]["balance"]
                print(f"Balance: Yes={balance['yes']} ({balance['ratio']:.2%}), No={balance['no']}")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"Generation complete!")
        print(f"Generated: {generated} samples")
        print(f"Failed: {failed} samples")
        print(f"Total dataset size: {len(self.dataset['samples'])} samples")
        
        balance = self.dataset["dataset_info"]["balance"]
        print(f"Final balance: Yes={balance['yes']} ({balance['ratio']:.2%}), No={balance['no']}")
        print(f"Dataset saved to: {self.output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate dataset incrementally with resume capability"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to generate in this run"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/api_dataset_incremental.json",
        help="Output dataset file (will append if exists)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("DEEPSEEK_API_KEY"),
        help="DeepSeek API key"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per sample"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        # Use the hardcoded key for convenience
        args.api_key = "sk-1a7afbc4eb614b56a1dfc74d69a6954c"
    
    # Create generator
    generator = IncrementalDatasetGenerator(args.api_key, args.output)
    
    # Generate samples
    generator.generate_samples(args.samples, args.max_retries)


if __name__ == "__main__":
    # Handle numpy import gracefully
    try:
        import numpy as np
    except ImportError:
        pass
    
    main()