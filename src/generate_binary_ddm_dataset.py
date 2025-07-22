#!/usr/bin/env python3
"""
Generate binary decision dataset for DDM (Drift Diffusion Model) training.

Creates a balanced dataset of Yes/No questions with reasoning traces
suitable for training models that predict decision dynamics.
"""
import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from r1_local_extractor import R1LocalExtractor
    from r1_extractor import R1Extractor
    from r1_api_adapter import R1APIAdapter
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    LOCAL_MODEL_AVAILABLE = False
    print("Warning: Local model not available. Using mock generation.")


class BinaryDDMDatasetGenerator:
    """Generate binary decision tasks with reasoning traces for DDM training."""
    
    def __init__(
        self,
        use_local_model: bool = True,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize dataset generator.
        
        Args:
            use_local_model: Use local model vs API
            model_name: Model name for local inference
            api_key: API key for remote inference
        """
        self.use_local_model = use_local_model
        self.extractor = None
        self.api_adapter = None
        
        if use_local_model and LOCAL_MODEL_AVAILABLE:
            try:
                self.extractor = R1LocalExtractor(model_name=model_name)
            except ImportError:
                print("Warning: Local model dependencies not installed, using mock generation")
        elif not use_local_model and api_key:
            # Use the new adapter for better API handling
            self.api_adapter = R1APIAdapter(api_key=api_key)
        else:
            print("Warning: No model available, using mock generation")
    
    def generate_math_verification_task(self, difficulty: str) -> Dict:
        """Generate a math verification task."""
        ranges = {
            "easy": (1, 50),
            "medium": (50, 200),
            "hard": (200, 1000)
        }
        
        min_val, max_val = ranges.get(difficulty, (1, 100))
        
        # Choose operation
        operation = random.choice(["+", "-", "×", "÷"])
        
        if operation == "+":
            a = random.randint(min_val, max_val)
            b = random.randint(min_val, max_val)
            correct_answer = a + b
            # Sometimes give wrong answer
            if random.random() < 0.5:
                given_answer = correct_answer
                is_correct = True
            else:
                given_answer = correct_answer + random.randint(-10, 10)
                if given_answer == correct_answer:
                    given_answer += 1
                is_correct = False
            prompt = f"Is {a} + {b} = {given_answer}?"
            
        elif operation == "-":
            a = random.randint(min_val, max_val)
            b = random.randint(0, a)  # Ensure positive result
            correct_answer = a - b
            if random.random() < 0.5:
                given_answer = correct_answer
                is_correct = True
            else:
                given_answer = correct_answer + random.randint(-10, 10)
                if given_answer == correct_answer:
                    given_answer += 1
                is_correct = False
            prompt = f"Is {a} - {b} = {given_answer}?"
            
        elif operation == "×":
            a = random.randint(min_val // 10, max_val // 10)
            b = random.randint(2, 20)
            correct_answer = a * b
            if random.random() < 0.5:
                given_answer = correct_answer
                is_correct = True
            else:
                given_answer = correct_answer + random.randint(-20, 20)
                if given_answer == correct_answer:
                    given_answer += 5
                is_correct = False
            prompt = f"Is {a} × {b} = {given_answer}?"
            
        else:  # division
            b = random.randint(2, 20)
            correct_answer = random.randint(min_val // 10, max_val // 10)
            a = b * correct_answer  # Ensure clean division
            if random.random() < 0.5:
                given_answer = correct_answer
                is_correct = True
            else:
                given_answer = correct_answer + random.randint(-5, 5)
                if given_answer == correct_answer:
                    given_answer += 1
                is_correct = False
            prompt = f"Is {a} ÷ {b} = {given_answer}?"
        
        return {
            "prompt": prompt,
            "category": "math_verification",
            "subcategory": f"arithmetic_{operation}",
            "difficulty": difficulty,
            "correct_answer": "Yes" if is_correct else "No",
            "metadata": {
                "operation": operation,
                "values": {"a": a, "b": b},
                "correct_result": correct_answer,
                "given_result": given_answer
            }
        }
    
    def generate_comparison_task(self, difficulty: str) -> Dict:
        """Generate a numerical comparison task."""
        ranges = {
            "easy": (1, 100),
            "medium": (100, 1000),
            "hard": (1000, 100000)
        }
        
        min_val, max_val = ranges.get(difficulty, (1, 100))
        
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        
        # Ensure they're different for medium/hard
        if difficulty != "easy" and a == b:
            b = a + random.randint(1, 10)
        
        comparison_type = random.choice(["greater", "less", "equal"])
        
        if comparison_type == "greater":
            prompt = f"Is {a} greater than {b}?"
            correct_answer = "Yes" if a > b else "No"
        elif comparison_type == "less":
            prompt = f"Is {a} less than {b}?"
            correct_answer = "Yes" if a < b else "No"
        else:  # equal
            # For equal, sometimes make them actually equal
            if random.random() < 0.3:
                b = a
            prompt = f"Is {a} equal to {b}?"
            correct_answer = "Yes" if a == b else "No"
        
        return {
            "prompt": prompt,
            "category": "comparison",
            "subcategory": f"numeric_{comparison_type}",
            "difficulty": difficulty,
            "correct_answer": correct_answer,
            "metadata": {
                "comparison_type": comparison_type,
                "values": {"a": a, "b": b}
            }
        }
    
    def generate_prime_check_task(self, difficulty: str) -> Dict:
        """Generate a prime number checking task."""
        ranges = {
            "easy": (2, 20),
            "medium": (20, 100),
            "hard": (100, 500)
        }
        
        min_val, max_val = ranges.get(difficulty, (2, 100))
        
        # List of primes up to 500
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
                 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
                 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241,
                 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
                 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389,
                 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461,
                 463, 467, 479, 487, 491, 499]
        
        # Filter primes in range
        primes_in_range = [p for p in primes if min_val <= p <= max_val]
        
        # 50/50 chance of prime vs non-prime
        if random.random() < 0.5 and primes_in_range:
            n = random.choice(primes_in_range)
            is_prime = True
        else:
            # Generate a composite number
            n = random.randint(min_val, max_val)
            while n in primes:
                n = random.randint(min_val, max_val)
            is_prime = False
        
        prompt = f"Is {n} a prime number?"
        correct_answer = "Yes" if is_prime else "No"
        
        return {
            "prompt": prompt,
            "category": "number_property",
            "subcategory": "prime_check",
            "difficulty": difficulty,
            "correct_answer": correct_answer,
            "metadata": {
                "number": n,
                "is_prime": is_prime
            }
        }
    
    def generate_divisibility_task(self, difficulty: str) -> Dict:
        """Generate a divisibility checking task."""
        divisors = {
            "easy": [2, 5, 10],
            "medium": [3, 4, 6, 8, 9],
            "hard": [7, 11, 13, 17]
        }
        
        divisor = random.choice(divisors.get(difficulty, [2, 3, 5]))
        
        # Generate number
        if difficulty == "easy":
            base = random.randint(1, 50)
        elif difficulty == "medium":
            base = random.randint(10, 200)
        else:
            base = random.randint(50, 500)
        
        # 50/50 chance of divisible vs not
        if random.random() < 0.5:
            n = base * divisor
            is_divisible = True
        else:
            n = base * divisor + random.randint(1, divisor - 1)
            is_divisible = False
        
        prompt = f"Is {n} divisible by {divisor}?"
        correct_answer = "Yes" if is_divisible else "No"
        
        return {
            "prompt": prompt,
            "category": "number_property",
            "subcategory": "divisibility",
            "difficulty": difficulty,
            "correct_answer": correct_answer,
            "metadata": {
                "number": n,
                "divisor": divisor,
                "is_divisible": is_divisible
            }
        }
    
    def generate_pattern_task(self, difficulty: str) -> Dict:
        """Generate a pattern matching task."""
        if difficulty == "easy":
            # Simple repetition
            patterns = ["AB", "ABC", "123", "XY"]
            pattern = random.choice(patterns)
            repetitions = random.randint(2, 3)
            
            if random.random() < 0.5:
                sequence = pattern * repetitions
                matches = True
            else:
                sequence = pattern * repetitions
                # Corrupt it
                idx = random.randint(0, len(sequence) - 1)
                sequence = sequence[:idx] + "?" + sequence[idx + 1:]
                matches = False
            
            prompt = f"Does '{sequence}' follow the pattern '{pattern}' repeated?"
            
        elif difficulty == "medium":
            # Arithmetic sequences
            start = random.randint(1, 20)
            diff = random.randint(2, 5)
            length = 5
            
            sequence = [start + i * diff for i in range(length)]
            
            if random.random() < 0.5:
                sequence_str = ", ".join(map(str, sequence))
                next_val = sequence[-1] + diff
                matches = True
            else:
                sequence_str = ", ".join(map(str, sequence[:-1]))
                next_val = sequence[-1] + random.randint(1, 10)
                if next_val == sequence[-1] + diff:
                    next_val += 1
                matches = False
            
            prompt = f"In the sequence {sequence_str}, is the next number {next_val}?"
            
        else:  # hard
            # Fibonacci-like
            a, b = random.randint(1, 5), random.randint(1, 5)
            sequence = [a, b]
            for _ in range(3):
                sequence.append(sequence[-1] + sequence[-2])
            
            if random.random() < 0.5:
                next_val = sequence[-1] + sequence[-2]
                matches = True
            else:
                next_val = sequence[-1] + sequence[-2] + random.randint(1, 5)
                matches = False
            
            sequence_str = ", ".join(map(str, sequence))
            prompt = f"In the sequence {sequence_str}, is the next number {next_val}?"
        
        correct_answer = "Yes" if matches else "No"
        
        return {
            "prompt": prompt,
            "category": "pattern_recognition",
            "subcategory": "sequence_matching",
            "difficulty": difficulty,
            "correct_answer": correct_answer,
            "metadata": {
                "pattern_type": difficulty,
                "matches": matches
            }
        }
    
    def generate_reasoning_trace(self, task: Dict) -> Dict:
        """Generate or extract reasoning trace for a task."""
        prompt = task["prompt"]
        
        # Use API adapter if available
        if hasattr(self, 'api_adapter') and self.api_adapter:
            try:
                result = self.api_adapter.query_binary_decision(prompt)
                
                if result.get("success"):
                    return {
                        "reasoning_tokens": result["reasoning_tokens"],
                        "model_answer": result["answer"],
                        "generation_time": time.time() - time.time()  # Will be tracked by adapter
                    }
                else:
                    print(f"API error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"API adapter failed: {e}")
        
        # Use extractor if available (local model)
        elif self.extractor:
            try:
                result = self.extractor.query(
                    prompt + "\nThink step by step and then answer with only 'Yes' or 'No'.",
                    temperature=0.7,
                    max_new_tokens=512
                )
                
                if "error" not in result:
                    reasoning_tokens = result.get("reasoning_tokens", [])
                    final_answer = result.get("final_answer", "").strip()
                    
                    # Extract Yes/No from final answer or reasoning
                    model_answer = "Unknown"
                    
                    # First check final answer
                    if final_answer:
                        if "yes" in final_answer.lower():
                            model_answer = "Yes"
                        elif "no" in final_answer.lower():
                            model_answer = "No"
                    
                    # If not found, check last reasoning token
                    if model_answer == "Unknown" and reasoning_tokens:
                        last_token = reasoning_tokens[-1].lower()
                        if "yes" in last_token:
                            model_answer = "Yes"
                        elif "no" in last_token:
                            model_answer = "No"
                    
                    return {
                        "reasoning_tokens": reasoning_tokens,
                        "model_answer": model_answer,
                        "generation_time": result.get("metadata", {}).get("generation_time", 0)
                    }
            except Exception as e:
                print(f"Model generation failed: {e}")
        
        # Fallback: Generate mock reasoning trace
        return self._generate_mock_reasoning(task)
    
    def _generate_mock_reasoning(self, task: Dict) -> Dict:
        """Generate mock reasoning trace for testing."""
        tokens = []
        
        if task["category"] == "math_verification":
            meta = task["metadata"]
            tokens.append(f"Let me check this calculation")
            tokens.append(f"I need to compute {meta['values']['a']} {meta['operation']} {meta['values']['b']}")
            tokens.append(f"The result is {meta['correct_result']}")
            tokens.append(f"The given answer is {meta['given_result']}")
            if meta['correct_result'] == meta['given_result']:
                tokens.append("These match")
                tokens.append("Yes, the equation is correct")
            else:
                tokens.append("These don't match")
                tokens.append("No, the equation is incorrect")
                
        elif task["category"] == "comparison":
            meta = task["metadata"]
            tokens.append(f"Let me compare these numbers")
            tokens.append(f"First number: {meta['values']['a']}")
            tokens.append(f"Second number: {meta['values']['b']}")
            if meta['comparison_type'] == "greater":
                if meta['values']['a'] > meta['values']['b']:
                    tokens.append(f"{meta['values']['a']} is indeed greater than {meta['values']['b']}")
                else:
                    tokens.append(f"{meta['values']['a']} is not greater than {meta['values']['b']}")
                    
        elif task["category"] == "number_property":
            if task["subcategory"] == "prime_check":
                n = task["metadata"]["number"]
                tokens.append(f"I need to check if {n} is prime")
                tokens.append(f"A prime number has no divisors except 1 and itself")
                if task["metadata"]["is_prime"]:
                    tokens.append(f"{n} has no divisors other than 1 and {n}")
                    tokens.append(f"Yes, {n} is prime")
                else:
                    tokens.append(f"{n} has other divisors")
                    tokens.append(f"No, {n} is not prime")
        
        # Add final answer
        tokens.append(task["correct_answer"])
        
        return {
            "reasoning_tokens": tokens,
            "model_answer": task["correct_answer"],
            "generation_time": random.uniform(0.5, 2.0)
        }
    
    def calculate_evidence_trajectory(self, tokens: List[str], correct_answer: str) -> Dict:
        """Calculate evidence accumulation trajectory from reasoning tokens."""
        evidence_scores = []
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            # Initial tokens usually neutral
            if i == 0:
                score = 0.0
            # Strong evidence tokens
            elif any(word in token_lower for word in ["therefore", "thus", "so", "yes", "no"]):
                score = 0.8 if correct_answer.lower() in token_lower else -0.8
            # Calculation/comparison tokens
            elif any(word in token_lower for word in ["equals", "is", "match", "correct", "incorrect"]):
                score = 0.6 if "not" not in token_lower else -0.6
            # Weak evidence
            else:
                score = random.uniform(-0.3, 0.3)
            
            evidence_scores.append(score)
        
        # Calculate cumulative evidence
        cumulative = []
        total = 0
        for score in evidence_scores:
            total += score
            cumulative.append(total)
        
        # Normalize timestamps
        total_time = len(tokens) * 0.3  # Assume 0.3s per token average
        timestamps = [i * (total_time / len(tokens)) for i in range(len(tokens))]
        
        # Determine if threshold was reached
        threshold = 1.5  # Decision threshold
        reached_threshold = abs(cumulative[-1]) >= threshold
        
        return {
            "evidence_scores": evidence_scores,
            "cumulative_evidence": cumulative,
            "timestamps": timestamps,
            "decision_time": timestamps[-1],
            "reached_threshold": reached_threshold,
            "final_evidence": cumulative[-1]
        }
    
    def generate_dataset(
        self,
        num_samples: int,
        categories: List[str],
        difficulty_distribution: Dict[str, float],
        output_path: str
    ) -> Dict:
        """
        Generate complete binary decision dataset.
        
        Args:
            num_samples: Number of samples to generate
            categories: List of task categories to include
            difficulty_distribution: Distribution of difficulties
            output_path: Path to save dataset
            
        Returns:
            Generated dataset
        """
        # Task generators by category
        generators = {
            "math": self.generate_math_verification_task,
            "comparison": self.generate_comparison_task,
            "prime": self.generate_prime_check_task,
            "divisibility": self.generate_divisibility_task,
            "pattern": self.generate_pattern_task
        }
        
        # Filter to requested categories
        active_generators = {k: v for k, v in generators.items() if k in categories}
        
        if not active_generators:
            raise ValueError(f"No valid categories found. Available: {list(generators.keys())}")
        
        # Initialize dataset
        dataset = {
            "dataset_info": {
                "version": "1.0",
                "task_type": "binary_decision",
                "total_samples": num_samples,
                "categories": categories,
                "difficulty_distribution": difficulty_distribution,
                "generation_date": datetime.now().isoformat()
            },
            "samples": []
        }
        
        # Track balance
        yes_count = 0
        no_count = 0
        
        print(f"Generating {num_samples} binary decision tasks...")
        
        for i in range(num_samples):
            # Choose category
            category = random.choice(list(active_generators.keys()))
            
            # Choose difficulty based on distribution
            difficulty = random.choices(
                list(difficulty_distribution.keys()),
                weights=list(difficulty_distribution.values())
            )[0]
            
            # Generate task
            task = active_generators[category](difficulty)
            
            # Generate reasoning trace
            print(f"[{i+1}/{num_samples}] Generating reasoning for: {task['prompt'][:50]}...")
            reasoning_result = self.generate_reasoning_trace(task)
            
            # Calculate evidence trajectory
            trajectory = self.calculate_evidence_trajectory(
                reasoning_result["reasoning_tokens"],
                task["correct_answer"]
            )
            
            # Create sample
            sample = {
                "id": f"binary_{i:05d}",
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
            
            dataset["samples"].append(sample)
            
            # Track balance
            if task["correct_answer"] == "Yes":
                yes_count += 1
            else:
                no_count += 1
            
            # Save periodically
            if (i + 1) % 10 == 0:
                self._save_dataset(dataset, output_path + ".tmp")
        
        # Update dataset info
        dataset["dataset_info"]["balance"] = {
            "yes": yes_count,
            "no": no_count,
            "ratio": yes_count / (yes_count + no_count)
        }
        
        # Save final dataset
        self._save_dataset(dataset, output_path)
        
        print(f"\nDataset generated successfully!")
        print(f"Total samples: {num_samples}")
        print(f"Yes answers: {yes_count} ({yes_count/num_samples*100:.1f}%)")
        print(f"No answers: {no_count} ({no_count/num_samples*100:.1f}%)")
        print(f"Saved to: {output_path}")
        
        return dataset
    
    def _save_dataset(self, dataset: Dict, path: str):
        """Save dataset to JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate binary decision dataset for DDM training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["math", "comparison", "prime"],
        choices=["math", "comparison", "prime", "divisibility", "pattern"],
        help="Task categories to include"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy:0.3,medium:0.5,hard:0.2",
        help="Difficulty distribution (format: easy:0.3,medium:0.5,hard:0.2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/binary_ddm_dataset.json",
        help="Output file path"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use API instead of local model"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or set DEEPSEEK_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Local model name"
    )
    
    args = parser.parse_args()
    
    # Parse difficulty distribution
    difficulty_dist = {}
    for item in args.difficulty.split(","):
        level, weight = item.split(":")
        difficulty_dist[level] = float(weight)
    
    # Normalize weights
    total_weight = sum(difficulty_dist.values())
    difficulty_dist = {k: v/total_weight for k, v in difficulty_dist.items()}
    
    # Create generator
    generator = BinaryDDMDatasetGenerator(
        use_local_model=not args.use_api,
        model_name=args.model,
        api_key=args.api_key or os.getenv("DEEPSEEK_API_KEY")
    )
    
    # Generate dataset
    dataset = generator.generate_dataset(
        num_samples=args.num_samples,
        categories=args.categories,
        difficulty_distribution=difficulty_dist,
        output_path=args.output
    )


if __name__ == "__main__":
    main()