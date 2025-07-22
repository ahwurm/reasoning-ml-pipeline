#!/usr/bin/env python3
"""Simple dataset validation without external dependencies."""
import json
from pathlib import Path
from collections import Counter

def validate_dataset(dataset_path: str):
    """Validate dataset structure and statistics."""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print("="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    
    # Basic info
    print(f"\nDataset: {dataset_path}")
    print(f"Total samples: {len(dataset['samples'])}")
    
    # Check structure
    print("\nSTRUCTURE CHECK:")
    if "dataset_info" in dataset:
        print("✓ dataset_info present")
        info = dataset["dataset_info"]
        print(f"  - Version: {info.get('version', 'N/A')}")
        print(f"  - Task type: {info.get('task_type', 'N/A')}")
        print(f"  - Categories: {info.get('categories', [])}")
    else:
        print("✗ dataset_info missing")
    
    if "samples" in dataset and isinstance(dataset["samples"], list):
        print(f"✓ samples present ({len(dataset['samples'])} items)")
    else:
        print("✗ samples missing or invalid")
        return
    
    # Analyze samples
    print("\nSAMPLE ANALYSIS:")
    
    # Required fields
    required_fields = ["id", "prompt", "category", "correct_answer", "reasoning_trace", "ddm_trajectory"]
    missing_fields = Counter()
    
    # Answer distribution
    answers = Counter()
    categories = Counter()
    difficulties = Counter()
    
    # Reasoning stats
    token_counts = []
    empty_reasoning = 0
    
    for i, sample in enumerate(dataset["samples"]):
        # Check fields
        for field in required_fields:
            if field not in sample:
                missing_fields[field] += 1
        
        # Count answers
        answers[sample.get("correct_answer", "Unknown")] += 1
        categories[sample.get("category", "Unknown")] += 1
        difficulties[sample.get("difficulty", "Unknown")] += 1
        
        # Check reasoning
        if "reasoning_trace" in sample:
            tokens = sample["reasoning_trace"].get("tokens", [])
            token_counts.append(len(tokens))
            if not tokens:
                empty_reasoning += 1
    
    # Report findings
    if missing_fields:
        print("\nMissing fields:")
        for field, count in missing_fields.items():
            print(f"  - {field}: {count} samples")
    else:
        print("✓ All required fields present")
    
    print("\nANSWER DISTRIBUTION:")
    total = sum(answers.values())
    for answer, count in answers.items():
        percent = (count / total * 100) if total > 0 else 0
        print(f"  {answer}: {count} ({percent:.1f}%)")
    
    print("\nCATEGORY DISTRIBUTION:")
    for category, count in categories.most_common():
        print(f"  {category}: {count}")
    
    print("\nDIFFICULTY DISTRIBUTION:")
    for difficulty, count in difficulties.most_common():
        print(f"  {difficulty}: {count}")
    
    if token_counts:
        print("\nREASONING STATISTICS:")
        print(f"  Average tokens: {sum(token_counts) / len(token_counts):.1f}")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
        print(f"  Empty reasoning: {empty_reasoning}")
    
    # Check balance
    print("\nBALANCE CHECK:")
    if "Yes" in answers and "No" in answers:
        yes_ratio = answers["Yes"] / (answers["Yes"] + answers["No"])
        balance_score = 1 - abs(0.5 - yes_ratio) * 2
        print(f"  Yes/No ratio: {yes_ratio:.2f}")
        print(f"  Balance score: {balance_score:.2f} (1.0 = perfect)")
        if balance_score > 0.8:
            print("  ✓ Good balance")
        else:
            print("  ⚠ Consider improving balance")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        validate_dataset(sys.argv[1])
    else:
        print("Usage: python simple_validate.py <dataset.json>")