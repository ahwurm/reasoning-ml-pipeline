#!/usr/bin/env python3
"""Debug reasoning generation."""
import json
import os
from generate_reasoning_traces import ReasoningGenerator

# Load datasets
with open("data/debates_reasoning.json", 'r') as f:
    existing = json.load(f)

with open("data/debates_1000_final.json", 'r') as f:
    full = json.load(f)

existing_ids = {q['id'] for q in existing['questions']}
print(f"Existing: {len(existing_ids)} questions")

# Find next questions
next_questions = []
for q in full['questions']:
    if q['id'] not in existing_ids:
        next_questions.append(q)
        if len(next_questions) >= 3:
            break

print(f"\nNext 3 questions to process:")
for q in next_questions:
    print(f"- {q['id']}: {q['question'][:60]}...")

# Try to generate reasoning for one
if next_questions:
    print(f"\nGenerating reasoning for {next_questions[0]['id']}...")
    generator = ReasoningGenerator(api_key=os.getenv("DEEPSEEK_API_KEY"))
    
    try:
        result = generator.generate_reasoning(next_questions[0]['question'], next_questions[0]['id'])
        print(f"Success! Answer: {result['answer']}, Confidence: {result['confidence']:.2f}")
    except Exception as e:
        print(f"Error: {e}")