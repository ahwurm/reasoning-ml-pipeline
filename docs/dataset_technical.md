# Technical Documentation: Viral Debates Dataset

## Dataset Structure

### JSON Schema
```json
{
  "metadata": {
    "total_questions": 1000,
    "base_questions": 100,
    "viral_debates": 80,
    "qc_controls": 20,
    "variations_per_question": 5,
    "polarities": ["positive", "negative"],
    "qc_expected_answers": {
      "Is Brad Pitt a famous actor?": "yes",
      ...
    }
  },
  "questions": [
    {
      "id": "q001_v1_pos",
      "base_id": "q001",
      "base_question": "Is a hot dog a sandwich?",
      "variation_num": 1,
      "polarity": "positive",
      "question": "Is a hot dog a sandwich?",
      "is_qc": false,
      "expected_answer": null  // only for QC questions
    },
    ...
  ]
}
```

## Variation Generation Algorithm

### Question Classification
Questions are classified into types for appropriate variation templates:

1. **Taxonomy** (`Is X a Y?`)
2. **Comparison** (`Are there more X than Y?`)
3. **Battle** (`Would X beat Y?`)
4. **Property** (`Is X wet?`)
5. **Action** (`Do people sleep?`)

### Template Application

#### Taxonomy Templates
```python
[
    "{subject} {verb} {object}",
    "Would you consider {subject} to be {object}",
    "Does {subject} qualify as {object}",
    "Can {subject} be classified as {object}",
    "Is {subject} technically {object}"
]
```

#### Comparison Templates
```python
[
    "Are there more {X} than {Y} in the world",
    "Do {X} outnumber {Y}",
    "Is the number of {X} greater than {Y}",
    "Are {X} more numerous than {Y}",
    "Does the world have more {X} than {Y}"
]
```

## Negation Strategies

### Type-Specific Negations

1. **Semantic Opposites** (Preferred)
   - `more → fewer`
   - `wet → dry`
   - `even → odd`
   - `beat → lose to`

2. **Classification Negations**
   - `"Does X qualify as Y?"` → `"Does X fail to qualify as Y?"`
   - `"Can X be classified as Y?"` → `"Is it wrong to classify X as Y?"`

3. **Clear Disagreement Forms**
   - `"Would you consider..."` → `"Do you disagree that..."`
   - `"Would you say..."` → `"Is it false that..."`

### Avoided Patterns
- Double negatives: `"not not"`
- Ambiguous constructions: `"Wouldn't you consider"`
- Passive voice: `"be defeated by"`
- Grammar errors: `"Is the not"`

## Quality Control Metrics

### Linguistic Confusion Detection
```python
problems_checked = {
    'would_wouldnt': "Wouldn't you" patterns,
    'double_negative': "not not" occurrences,
    'grammar_errors': "Is the not" constructions,
    'contradictions': "fewer numerous",
    'passive_voice': "be defeated by",
    'awkward_negation': "not true that"
}
```

### Resolution Process
1. **Pattern Detection**: Scan for problematic patterns
2. **Rule-Based Fixes**: Apply specific replacements
3. **Semantic Negation**: Use opposites where available
4. **Grammar Cleanup**: Fix word order and structure
5. **Validation**: Ensure 0% confusion rate

## Binary Answer Mapping

### Converted Questions
For questions originally phrased as choices:

| Original | Converted | Yes = | No = |
|----------|-----------|-------|------|
| "Are there more wheels or doors?" | "Are there more wheels than doors?" | more wheels | more doors |
| "Is math invented or discovered?" | "Is math invented?" | invented | discovered |
| "Is zero even or odd?" | "Is zero even?" | even | odd |

### QC Answer Tracking
All QC questions have known expected answers:
- Stored in metadata for validation
- Negations flip the expected answer
- Used to verify model sanity

## Performance Considerations

### Efficient Processing
- Questions grouped by base_id for batch processing
- Variations generated systematically
- Linguistic fixes applied in single pass

### Memory Usage
- 1000 questions ≈ 500KB JSON
- Structured for easy filtering/querying
- Indexed by multiple fields (id, base_id, type)

## Usage Examples

### Loading Dataset
```python
import json

with open('data/debates_1000_final.json', 'r') as f:
    dataset = json.load(f)

# Get all variations of a question
base_id = "q001"
variations = [q for q in dataset['questions'] if q['base_id'] == base_id]

# Get only QC questions
qc_questions = [q for q in dataset['questions'] if q['is_qc']]

# Get positive/negative pairs
for q in dataset['questions']:
    if q['polarity'] == 'positive':
        neg_id = q['id'].replace('_pos', '_neg')
        neg_q = next((nq for nq in dataset['questions'] if nq['id'] == neg_id), None)
```

### Analyzing Consistency
```python
# Group by base question
from collections import defaultdict

responses_by_base = defaultdict(list)
for q in questions:
    answer = model.predict(q['question'])
    responses_by_base[q['base_id']].append({
        'variation': q['variation_num'],
        'polarity': q['polarity'],
        'answer': answer
    })

# Check consistency
for base_id, responses in responses_by_base.items():
    positive_answers = [r['answer'] for r in responses if r['polarity'] == 'positive']
    if len(set(positive_answers)) > 1:
        print(f"Inconsistent answers for {base_id}")
```

## Edge Cases Handled

1. **Food items with "hot"**: Don't apply hot→cold antonym
2. **Complex comparisons**: Maintain grammatical structure
3. **Rhetorical questions**: Convert to statements
4. **Multiple valid negations**: Choose most natural
5. **Context-dependent words**: Preserve original meaning