# Viral Debates Dataset Methodology

## Overview

This dataset contains 1000 yes/no questions designed to test model reasoning consistency on genuinely ambiguous topics. Unlike traditional benchmarks that focus on factual accuracy or ethical dilemmas, this dataset captures the essence of viral internet debates - questions that spark endless arguments because they have no clear answer.

## Why Viral Debates?

We chose viral debate questions (like "Is a hot dog a sandwich?") because they:

1. **Genuine Ambiguity**: No objectively correct answer exists
2. **Reasoning-Heavy**: Require weighing multiple valid perspectives  
3. **Culturally Relevant**: Based on actual debates people have
4. **Non-Political**: Avoid ethics/politics that might trigger safety filters
5. **Fun & Engaging**: More interesting than abstract logic puzzles

## Dataset Composition

### Base Questions (100 total)

#### 80 Viral Debates
Categories include:
- **Taxonomy Wars** (15): "Is a hot dog a sandwich?"
- **Impossible Counting** (10): "Are there more wheels or doors?"
- **Battle Scenarios** (10): "Would 100 humans beat a gorilla?"
- **Simple but Deep** (15): "Is water wet?"
- **Pop Culture** (15): "Is Die Hard a Christmas movie?"
- **Modern Life** (15): "Should pineapple go on pizza?"

#### 20 Quality Control (QC) Questions
Consensus questions requiring reasoning:
- **Cultural Icons** (5): "Is Brad Pitt a famous actor?"
- **Food Categories** (5): "Is pasta commonly eaten for dinner?"
- **Common Activities** (5): "Do people sleep at night?"
- **Social Norms** (5): "Is it polite to say thank you?"

### Variation Generation (10x per base question)

For each base question, we generate:
1. **5 Phrasings**: From casual to formal
   - Original: "Is a hot dog a sandwich?"
   - Formal: "Would you consider a hot dog to be a sandwich?"
   - Classification: "Does a hot dog qualify as a sandwich?"
   - Technical: "Can a hot dog be classified as a sandwich?"
   - Definitional: "Is a hot dog technically a sandwich?"

2. **2 Polarities**: Positive and negative versions
   - Positive: "Is a hot dog a sandwich?"
   - Negative: "Is a hot dog not a sandwich?"

Total: 100 base × 5 variations × 2 polarities = **1000 questions**

## Linguistic Clarity Improvements

### Initial Problems (24% of questions)
1. **Would/Wouldn't Ambiguity**: "Wouldn't you consider..." can mean agreement or disagreement
2. **Grammar Errors**: "Is the not number of wheels..."
3. **Double Negatives**: "Can X not not be Y?"
4. **Contradictions**: "Are X fewer numerous than Y?"

### Solutions Applied
1. **Semantic Negations**: Use opposites (wet→dry, more→fewer) instead of adding "not"
2. **Clear Disagreement**: "Wouldn't you..." → "Do you disagree that..."
3. **Active Voice**: "Would X be defeated by Y?" → "Would X lose to Y?"
4. **Natural Language**: Removed awkward constructions

Result: **0% linguistic confusion** in final dataset

## Why This Approach?

### For MC Dropout Training
- **Consistency Testing**: Same question, different phrasings should yield similar reasoning
- **Uncertainty Calibration**: Viral debates should show high uncertainty, QC should show low
- **Robustness**: Model should handle various phrasings without confusion

### For Research
- **Reasoning Patterns**: How do models approach ambiguous questions?
- **Linguistic Robustness**: Do slight rephrases change the answer?
- **Calibration**: Can models express appropriate uncertainty?

## Dataset Properties

- **Format**: JSON with clear structure
- **Size**: 1000 questions (manageable for experiments)
- **Balance**: 80% ambiguous, 20% control
- **Quality**: No linguistic confusion
- **Accessibility**: Based on popular culture, not specialized knowledge

## Next Steps

1. Generate reasoning traces using DeepSeek R1
2. Train MC Dropout models on reasoning + answers
3. Analyze consistency across variations
4. Measure uncertainty calibration on viral vs QC questions

This dataset provides a unique testbed for studying how AI systems handle genuine ambiguity - the kind humans argue about endlessly on the internet.