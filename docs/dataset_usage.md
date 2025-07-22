# Dataset Usage Guide

## Quick Start

### Loading the Dataset
```python
import json

# Load the dataset
with open('data/debates_1000_final.json', 'r') as f:
    data = json.load(f)

# Access questions
questions = data['questions']
metadata = data['metadata']

# Example question
print(questions[0])
# Output:
# {
#   "id": "q001_v1_pos",
#   "base_id": "q001", 
#   "base_question": "Is a hot dog a sandwich?",
#   "variation_num": 1,
#   "polarity": "positive",
#   "question": "Is a hot dog a sandwich?",
#   "is_qc": false
# }
```

## Understanding the Structure

### Question Fields
- **id**: Unique identifier (format: `q{base}_v{variation}_{polarity}`)
- **base_id**: Groups all 10 variations of same base question
- **base_question**: The original question
- **variation_num**: Which of the 5 phrasings (1-5)
- **polarity**: "positive" or "negative"
- **question**: The actual question text
- **is_qc**: Boolean indicating if it's a quality control question
- **expected_answer**: Only present for QC questions ("yes" or "no")

### Metadata Fields
- **total_questions**: 1000
- **base_questions**: 100
- **viral_debates**: 80
- **qc_controls**: 20
- **qc_expected_answers**: Dictionary of QC questions → answers

## Common Use Cases

### 1. Get All Variations of a Question
```python
def get_variations(data, base_question):
    return [q for q in data['questions'] 
            if q['base_question'] == base_question]

# Example
hotdog_variations = get_variations(data, "Is a hot dog a sandwich?")
for q in hotdog_variations:
    print(f"{q['polarity']:8} V{q['variation_num']}: {q['question']}")
```

### 2. Separate Viral Debates from QC Questions
```python
viral_questions = [q for q in questions if not q['is_qc']]
qc_questions = [q for q in questions if q['is_qc']]

print(f"Viral debates: {len(viral_questions)}")  # 800
print(f"QC questions: {len(qc_questions)}")      # 200
```

### 3. Get Positive/Negative Pairs
```python
def get_pairs(questions):
    pairs = []
    for q in questions:
        if q['polarity'] == 'positive':
            neg_id = q['id'].replace('_pos', '_neg')
            neg_q = next((nq for nq in questions if nq['id'] == neg_id), None)
            if neg_q:
                pairs.append((q, neg_q))
    return pairs

pairs = get_pairs(questions)
print(f"Found {len(pairs)} positive/negative pairs")
```

### 4. Evaluate QC Performance
```python
def evaluate_qc(model, data):
    qc_questions = [q for q in data['questions'] if q['is_qc']]
    correct = 0
    
    for q in qc_questions:
        prediction = model.predict(q['question'])  # Should return "yes" or "no"
        if prediction == q['expected_answer']:
            correct += 1
    
    accuracy = correct / len(qc_questions)
    print(f"QC Accuracy: {accuracy:.2%}")
    return accuracy
```

### 5. Analyze Consistency Across Variations
```python
def analyze_consistency(model, data):
    from collections import defaultdict
    
    # Group responses by base question
    responses = defaultdict(lambda: {'positive': [], 'negative': []})
    
    for q in data['questions']:
        answer = model.predict(q['question'])
        responses[q['base_id']][q['polarity']].append(answer)
    
    # Check consistency
    inconsistent = []
    for base_id, pols in responses.items():
        # Check if all positive variations give same answer
        if len(set(pols['positive'])) > 1:
            inconsistent.append(base_id)
    
    print(f"Inconsistent bases: {len(inconsistent)}/{len(responses)}")
    return inconsistent
```

## Expected Model Behavior

### On Viral Debates
- **High uncertainty**: Model should express doubt
- **Reasonable arguments**: Can argue either side
- **Consistency**: Same answer across phrasings
- **Polarity awareness**: Negations should flip answer

### On QC Questions
- **High confidence**: Model should be certain
- **Correct answers**: Match expected_answer field
- **Consistency**: All variations same answer
- **Clear reasoning**: Straightforward logic

## Working with Reasoning

When adding reasoning traces:
```python
# Structure for reasoning dataset
reasoning_data = {
    "metadata": data['metadata'],
    "questions": []
}

for q in data['questions']:
    # Get reasoning from model
    reasoning = model.generate_reasoning(q['question'])
    
    q_with_reasoning = q.copy()
    q_with_reasoning['reasoning'] = reasoning
    q_with_reasoning['final_answer'] = extract_answer(reasoning)
    
    reasoning_data['questions'].append(q_with_reasoning)
```

## Tips for Training

1. **Batch by Base ID**: Process all variations together
2. **Balance Polarities**: Equal positive/negative examples  
3. **Monitor QC Performance**: Should stay near 100%
4. **Track Consistency**: Variations should align
5. **Measure Uncertainty**: Higher for viral, lower for QC

## Filtering Examples

```python
# Get only "hot dog" related questions
hotdog_qs = [q for q in questions if 'hot dog' in q['question']]

# Get only comparison questions
comparison_qs = [q for q in questions if 'more' in q['question'] and 'than' in q['question']]

# Get first variation of each base
first_variations = [q for q in questions if q['variation_num'] == 1]

# Get all negative polarity questions
negative_qs = [q for q in questions if q['polarity'] == 'negative']
```

## Error Handling

```python
# Safe QC answer checking
def get_expected_answer(question):
    if question.get('is_qc', False):
        return question.get('expected_answer', 'unknown')
    return None  # Viral debates have no expected answer

# Validate dataset integrity
def validate_dataset(data):
    issues = []
    
    # Check all base questions have 10 variations
    from collections import Counter
    base_counts = Counter(q['base_id'] for q in data['questions'])
    for base_id, count in base_counts.items():
        if count != 10:
            issues.append(f"{base_id} has {count} variations (expected 10)")
    
    # Check QC questions have expected answers
    for q in data['questions']:
        if q['is_qc'] and 'expected_answer' not in q:
            issues.append(f"{q['id']} is QC but missing expected_answer")
    
    return issues
```

## Next Steps

1. Generate reasoning traces for each question
2. Train models on question + reasoning → answer
3. Evaluate consistency and uncertainty
4. Analyze which variations cause most disagreement
5. Study how reasoning changes with phrasing