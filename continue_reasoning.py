#!/usr/bin/env python3
"""Continue reasoning generation with better error handling and progress tracking."""
import json
import time
from pathlib import Path
from generate_reasoning_traces import ReasoningGenerator
import os

def main():
    output_path = Path("data/debates_reasoning.json")
    
    # Load existing progress
    with open(output_path, 'r') as f:
        existing_data = json.load(f)
    
    existing_ids = {q['id'] for q in existing_data['questions']}
    print(f"Loaded {len(existing_ids)} existing results")
    
    # Load full dataset
    with open("data/debates_1000_final.json", 'r') as f:
        full_dataset = json.load(f)
    
    # Find remaining questions
    remaining = [q for q in full_dataset['questions'] if q['id'] not in existing_ids]
    print(f"Remaining questions: {len(remaining)}")
    
    if not remaining:
        print("All questions already processed!")
        return
    
    # Initialize generator
    generator = ReasoningGenerator(api_key=os.getenv("DEEPSEEK_API_KEY"))
    
    # Process remaining questions
    batch_size = 5
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(existing_data['questions']) + i + 1}-{len(existing_data['questions']) + i + len(batch)} / 1000)")
        
        for question in batch:
            try:
                # Generate reasoning
                result = generator.generate_reasoning(question['question'], question['id'])
                
                # Add to existing data
                existing_data['questions'].append({
                    'id': question['id'],
                    'question': question['question'],
                    'is_qc': question['is_qc'],
                    'base_id': question['base_id'],
                    'variation': question['variation'],
                    'answer': result.answer,
                    'reasoning': result.reasoning,
                    'confidence': result.confidence
                })
                
                print(f"✓ {question['id']}: {result.answer} (confidence: {result.confidence:.2f})")
                
                # Save after each question
                with open(output_path, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                
            except Exception as e:
                print(f"✗ {question['id']}: Error - {str(e)}")
                time.sleep(5)  # Wait before retry
        
        # Pause between batches
        if i + batch_size < len(remaining):
            print(f"Saved {len(existing_data['questions'])} total. Pausing...")
            time.sleep(10)
    
    print(f"\n✓ Completed! Total questions: {len(existing_data['questions'])}")

if __name__ == "__main__":
    main()