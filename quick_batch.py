#!/usr/bin/env python3
"""Quick batch runner - runs a single batch and exits."""
import subprocess
import sys

def run_batch(size=30):
    """Run a single batch of generation."""
    cmd = [
        sys.executable,
        "incremental_generate.py",
        "--samples", str(size),
        "--output", "data/api_dataset_incremental.json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    # Extract final count from output
    output_lines = result.stdout.strip().split('\n')
    for line in output_lines:
        if "Total dataset size:" in line:
            count = int(line.split(":")[1].split()[0])
            print(f"Total samples after batch: {count}")
            return True
    
    return True

if __name__ == "__main__":
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_batch(batch_size)