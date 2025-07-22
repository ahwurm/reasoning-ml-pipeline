#!/usr/bin/env python3
"""
Batch runner for reasoning generation.
Processes questions in batches with configurable size.
"""
import subprocess
import json
import sys
import time
from datetime import datetime

def get_current_progress():
    """Get current number of processed questions."""
    try:
        with open('data/aw_debates_with_reasoning.json', 'r') as f:
            data = json.load(f)
            return len(data['questions'])
    except:
        return 0

def run_batch(batch_size=10, total_questions=1000):
    """Run a single batch of reasoning generation."""
    current = get_current_progress()
    
    if current >= total_questions:
        print(f"✓ All {total_questions} questions have been processed!")
        return False
    
    # Calculate limit for this batch
    limit = min(current + batch_size, total_questions)
    
    print(f"\n{'='*60}")
    print(f"Batch Processing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Current progress: {current}/{total_questions} ({current/total_questions*100:.1f}%)")
    print(f"Processing batch: questions {current+1} to {limit}")
    print(f"Batch size: {limit - current} questions")
    print(f"{'='*60}\n")
    
    # Run the generation script
    cmd = [
        'python', 'generate_reasoning_traces.py',
        '--input', 'data/aw_debates_dataset.json',
        '--output', 'data/aw_debates_with_reasoning.json',
        '--batch-size', '1',
        '--delay', '2.0',
        '--model', 'deepseek-reasoner',
        '--limit', str(limit)
    ]
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        
        # Check new progress
        new_current = get_current_progress()
        processed = new_current - current
        
        print(f"\n{'='*60}")
        print(f"✓ Batch completed successfully!")
        print(f"  Processed: {processed} questions")
        print(f"  Time: {elapsed/60:.1f} minutes")
        print(f"  Rate: {elapsed/processed:.1f} seconds/question" if processed > 0 else "")
        print(f"  Total progress: {new_current}/{total_questions} ({new_current/total_questions*100:.1f}%)")
        print(f"{'='*60}\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Batch failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Batch interrupted by user")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run reasoning generation in batches")
    parser.add_argument('--batch-size', type=int, default=10, 
                       help='Number of questions per batch (default: 10)')
    parser.add_argument('--total', type=int, default=1000,
                       help='Total questions in dataset (default: 1000)')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuously until all questions are processed')
    
    args = parser.parse_args()
    
    if args.continuous:
        print("Running in continuous mode. Press Ctrl+C to stop.")
        batch_count = 0
        
        while True:
            if not run_batch(args.batch_size, args.total):
                break
                
            batch_count += 1
            
            # Check if we've completed 10 batches
            if batch_count >= 10:
                response = input("\n✓ Completed 10 batches. Continue? [y]/n: ").strip().lower()
                if response == 'n':
                    print("Stopping as requested.")
                    break
                batch_count = 0  # Reset counter
                print("Continuing with next 10 batches...")
            
            print("Waiting 10 seconds before next batch...")
            time.sleep(10)
    else:
        run_batch(args.batch_size, args.total)