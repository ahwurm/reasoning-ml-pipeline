#!/usr/bin/env python3
"""
Generate reasoning traces for viral debates dataset using DeepSeek R1.
Supports incremental processing and resume functionality.
"""

import json
import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import logging
import shutil
import glob
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result from reasoning generation."""
    question_id: str
    question: str
    reasoning: str
    answer: str
    reasoning_tokens: List[str]
    confidence: Optional[float] = None
    error: Optional[str] = None
    truncated: bool = False


class ReasoningGenerator:
    """Generate reasoning traces using DeepSeek R1."""
    
    def __init__(self, api_key: str, model: str = "deepseek-reasoner"):
        """Initialize the generator."""
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model = model
        
    def generate_reasoning(self, question: str, question_id: str) -> ReasoningResult:
        """Generate reasoning for a single question."""
        # Log model being used (only once)
        if not hasattr(self, '_logged_model'):
            logger.info(f"API calls will use model: {self.model}")
            self._logged_model = True
            
        prompt = f"""Question: {question}

Please think through this step-by-step, considering multiple perspectives and arguments.
For ambiguous questions, acknowledge the ambiguity and explain both sides.
Provide your detailed reasoning process, then conclude with your answer.

Your final answer must be exactly "yes" or "no" on a new line after your reasoning."""

        try:
            # Call DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful reasoner who thinks through questions systematically."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=8000  # Maximum allowed by DeepSeek API
            )
            
            # Extract content and check for truncation
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == "length":
                logger.warning(f"Response truncated at 8000 tokens for {question_id}")
            elif finish_reason != "stop":
                logger.warning(f"Unexpected finish_reason '{finish_reason}' for {question_id}")
            
            # Parse reasoning and answer
            lines = content.strip().split('\n')
            
            # Find the answer (should be last line)
            answer = None
            reasoning_lines = []
            
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip().lower()
                if line in ["yes", "no"]:
                    answer = line
                    reasoning_lines = lines[:i]
                    break
            
            if not answer:
                # Try to find yes/no anywhere in the last few lines
                for line in reversed(lines[-3:]):
                    if "yes" in line.lower() and "no" not in line.lower():
                        answer = "yes"
                        break
                    elif "no" in line.lower() and "yes" not in line.lower():
                        answer = "no"
                        break
                
                if not answer:
                    logger.warning(f"No clear answer found for {question_id}")
                    answer = "unknown"
                
                reasoning_lines = lines
            
            reasoning = '\n'.join(reasoning_lines).strip()
            
            # Tokenize reasoning (simple word tokenization for now)
            reasoning_tokens = reasoning.split()
            
            # Estimate confidence based on reasoning characteristics
            confidence = self._estimate_confidence(reasoning, answer)
            
            return ReasoningResult(
                question_id=question_id,
                question=question,
                reasoning=reasoning,
                answer=answer,
                reasoning_tokens=reasoning_tokens,
                confidence=confidence,
                truncated=(finish_reason == "length")
            )
            
        except Exception as e:
            logger.error(f"Error generating reasoning for {question_id}: {str(e)}")
            return ReasoningResult(
                question_id=question_id,
                question=question,
                reasoning="",
                answer="unknown",
                reasoning_tokens=[],
                error=str(e)
            )
    
    def _estimate_confidence(self, reasoning: str, answer: str) -> float:
        """Estimate confidence based on reasoning patterns."""
        if answer == "unknown":
            return 0.0
            
        # Use Token-Level Context (TLC) method
        reasoning_tokens = reasoning.split()
        return self._calculate_tlc_confidence(reasoning_tokens, answer, reasoning)
    
    def _calculate_tlc_confidence(self, reasoning_tokens: List[str], answer: str, full_reasoning: str) -> float:
        """Calculate confidence using Token-Level Context (TLC) method."""
        import numpy as np
        
        if not reasoning_tokens:
            return 0.5
        
        # Token categories with importance weights
        HIGH_IMPORTANCE_TOKENS = {
            # Conclusion markers (weight: 1.0)
            "therefore": 1.0, "thus": 1.0, "conclude": 1.0, "consequently": 1.0,
            "hence": 1.0, "accordingly": 1.0, "ergo": 1.0,
            
            # Strong evidence markers (weight: 0.8)
            "proves": 0.8, "demonstrates": 0.8, "establishes": 0.8, "confirms": 0.8,
            "evidence": 0.8, "definitively": 0.8, "unquestionably": 0.8,
            
            # Certainty markers (weight: 0.7)
            "clearly": 0.7, "definitely": 0.7, "obviously": 0.7, "certainly": 0.7,
            "absolutely": 0.7, "undoubtedly": 0.7
        }
        
        UNCERTAINTY_TOKENS = {
            # Strong uncertainty (weight: -0.8)
            "however": -0.8, "although": -0.8, "nevertheless": -0.8, "nonetheless": -0.8,
            
            # Ambiguity markers (weight: -0.7)
            "ambiguous": -0.7, "unclear": -0.7, "debatable": -0.7, "controversial": -0.7,
            "subjective": -0.7, "depends": -0.7,
            
            # Weak uncertainty (weight: -0.5)
            "possibly": -0.5, "perhaps": -0.5, "maybe": -0.5, "might": -0.5,
            "could": -0.5, "potentially": -0.5, "arguably": -0.5
        }
        
        # Directional tokens based on answer
        if answer.lower() == "yes":
            DIRECTIONAL_TOKENS = {
                "yes": 0.6, "true": 0.6, "correct": 0.6, "affirm": 0.6, "positive": 0.6,
                "no": -0.6, "false": -0.6, "incorrect": -0.6, "negative": -0.6, "not": -0.3
            }
        else:  # answer is "no"
            DIRECTIONAL_TOKENS = {
                "no": 0.6, "false": 0.6, "incorrect": 0.6, "negative": 0.6, "not": 0.3,
                "yes": -0.6, "true": -0.6, "correct": -0.6, "affirm": -0.6, "positive": -0.6
            }
        
        # Position-based weighting (exponential increase toward end)
        # Later tokens in reasoning have more weight
        position_weights = np.exp(np.linspace(0, 2, len(reasoning_tokens)))
        position_weights = position_weights / position_weights.sum()
        
        # Calculate token scores
        token_scores = []
        for i, token in enumerate(reasoning_tokens):
            token_lower = token.lower().strip('.,!?;:')
            
            # Get token importance
            importance = 0.0
            
            # Check each category
            if token_lower in HIGH_IMPORTANCE_TOKENS:
                importance += HIGH_IMPORTANCE_TOKENS[token_lower]
            
            if token_lower in UNCERTAINTY_TOKENS:
                importance += UNCERTAINTY_TOKENS[token_lower]
            
            if token_lower in DIRECTIONAL_TOKENS:
                importance += DIRECTIONAL_TOKENS[token_lower]
            
            # Apply position weight
            weighted_score = importance * position_weights[i]
            token_scores.append(weighted_score)
        
        # Calculate overall confidence
        # Base confidence of 0.5
        base_confidence = 0.5
        
        # Sum weighted token scores
        total_score = sum(token_scores)
        
        # Normalize by number of tokens (prevent long reasoning from dominating)
        normalized_score = total_score / np.sqrt(len(reasoning_tokens))
        
        # Apply sigmoid-like transformation for smooth confidence values
        confidence = base_confidence + np.tanh(normalized_score * 2) * 0.45
        
        # Additional penalty for conflicting arguments
        reasoning_lower = full_reasoning.lower()
        if "on the other hand" in reasoning_lower and "on one hand" in reasoning_lower:
            confidence *= 0.8  # Reduce confidence for explicit two-sided arguments
        
        if reasoning_lower.count("however") > 2:
            confidence *= 0.9  # Reduce confidence for many contrasting points
        
        # Clamp to [0.1, 0.95]
        return max(0.1, min(0.95, confidence))


def load_progress(output_path: str) -> Dict[str, ReasoningResult]:
    """Load existing progress from output file."""
    if not os.path.exists(output_path):
        return {}
    
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
            
        # Convert back to ReasoningResult objects
        progress = {}
        for item in data.get('questions', []):
            progress[item['id']] = ReasoningResult(
                question_id=item['id'],
                question=item['question'],
                reasoning=item.get('reasoning', ''),
                answer=item.get('answer', 'unknown'),
                reasoning_tokens=item.get('reasoning_tokens', []),
                confidence=item.get('confidence'),
                error=item.get('error')
            )
        
        return progress
        
    except Exception as e:
        logger.error(f"Error loading progress: {str(e)}")
        return {}


def save_progress(results: List[Dict], metadata: Dict, output_path: str):
    """Save current progress to file with automatic backup."""
    # Create backup if file exists
    if os.path.exists(output_path):
        backup_path = f"{output_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(output_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        
        # Keep only last 3 backups
        backup_pattern = f"{output_path}.backup_*"
        backups = sorted(glob.glob(backup_pattern))
        if len(backups) > 3:
            for old_backup in backups[:-3]:
                os.remove(old_backup)
                logger.info(f"Removed old backup: {old_backup}")
    
    output = {
        "metadata": metadata,
        "questions": results
    }
    
    # Save with pretty formatting
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Saved progress: {len(results)} questions processed")


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning traces for debate questions")
    parser.add_argument(
        "--input",
        default="data/debates_1000_final.json",
        help="Input dataset path"
    )
    parser.add_argument(
        "--output",
        default="data/debates_reasoning.json",
        help="Output path for reasoning dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for saving progress"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of questions to process"
    )
    parser.add_argument(
        "--model",
        default="deepseek-reasoner",
        help="Model to use (deepseek-reasoner or deepseek-chat)"
    )
    
    args = parser.parse_args()
    
    # Load API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY not found in environment")
        return
    
    # Initialize generator
    generator = ReasoningGenerator(api_key, model=args.model)
    logger.info(f"Using model: {args.model}")
    
    # Load input dataset
    logger.info(f"Loading dataset from {args.input}")
    with open(args.input, 'r') as f:
        dataset = json.load(f)
    
    questions = dataset['questions']
    if args.limit:
        questions = questions[:args.limit]
    
    # Load existing progress
    progress = load_progress(args.output)
    logger.info(f"Loaded {len(progress)} existing results")
    
    # Prepare metadata
    metadata = dataset['metadata'].copy()
    metadata['reasoning_generation'] = {
        'model': generator.model,
        'timestamp': datetime.now().isoformat(),
        'total_questions': len(questions)
    }
    
    # Process questions
    results = []
    new_count = 0
    
    with tqdm(total=len(questions), desc="Generating reasoning") as pbar:
        for i, question_data in enumerate(questions):
            question_id = question_data['id']
            
            # Skip if already processed
            if question_id in progress:
                result = progress[question_id]
                results.append({
                    'id': result.question_id,
                    'base_id': question_data['base_id'],
                    'question': result.question,
                    'reasoning': result.reasoning,
                    'answer': result.answer,
                    'reasoning_tokens': result.reasoning_tokens,
                    'confidence': result.confidence,
                    'is_qc': question_data['is_qc'],
                    'expected_answer': question_data.get('expected_answer'),
                    'variation_num': question_data.get('variation_num', 1),
                    'polarity': question_data.get('polarity', 'unknown'),
                    'error': result.error
                })
                pbar.update(1)
                continue
            
            # Generate reasoning
            result = generator.generate_reasoning(
                question_data['question'],
                question_id
            )
            
            # Add to results
            results.append({
                'id': result.question_id,
                'base_id': question_data['base_id'],
                'question': result.question,
                'reasoning': result.reasoning,
                'answer': result.answer,
                'reasoning_tokens': result.reasoning_tokens,
                'confidence': result.confidence,
                'is_qc': question_data['is_qc'],
                'expected_answer': question_data.get('expected_answer'),
                'variation_num': question_data.get('variation_num', 1),
                'polarity': question_data.get('polarity', 'unknown'),
                'error': result.error,
                'truncated': getattr(result, 'truncated', False)
            })
            
            new_count += 1
            pbar.update(1)
            
            # Save progress periodically
            if new_count % args.batch_size == 0:
                save_progress(results, metadata, args.output)
            
            # Rate limiting
            if i < len(questions) - 1:  # Don't delay after last question
                time.sleep(args.delay)
    
    # Final save
    save_progress(results, metadata, args.output)
    
    # Print summary
    logger.info(f"\nProcessing complete!")
    logger.info(f"Total questions: {len(results)}")
    logger.info(f"New reasoning generated: {new_count}")
    
    # Analyze results
    qc_correct = 0
    qc_total = 0
    unknown_answers = 0
    
    for result in results:
        if result['answer'] == 'unknown':
            unknown_answers += 1
        
        if result['is_qc'] and result['expected_answer']:
            qc_total += 1
            if result['answer'] == result['expected_answer']:
                qc_correct += 1
    
    if qc_total > 0:
        logger.info(f"QC Accuracy: {qc_correct}/{qc_total} ({qc_correct/qc_total*100:.1f}%)")
    
    if unknown_answers > 0:
        logger.warning(f"Unknown answers: {unknown_answers}")
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()