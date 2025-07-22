#!/usr/bin/env python3
"""
Validate binary DDM dataset for quality and consistency.

Checks for:
- Data format correctness
- Answer balance (Yes/No ratio)
- Category distribution
- Reasoning trace quality
- DDM trajectory validity
"""
import argparse
import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class DDMDatasetValidator:
    """Validate DDM dataset quality and statistics."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize validator with dataset.
        
        Args:
            dataset_path: Path to dataset JSON file
        """
        self.dataset_path = Path(dataset_path)
        self.dataset = self._load_dataset()
        self.validation_results = {}
    
    def _load_dataset(self) -> Dict:
        """Load dataset from JSON file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def validate_structure(self) -> Dict:
        """Validate dataset structure and format."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check top-level structure
        required_fields = ["dataset_info", "samples"]
        for field in required_fields:
            if field not in self.dataset:
                results["errors"].append(f"Missing required field: {field}")
                results["valid"] = False
        
        if not results["valid"]:
            return results
        
        # Check dataset info
        info_fields = ["version", "task_type", "total_samples"]
        for field in info_fields:
            if field not in self.dataset["dataset_info"]:
                results["warnings"].append(f"Missing dataset_info field: {field}")
        
        # Check samples
        if not isinstance(self.dataset["samples"], list):
            results["errors"].append("Samples must be a list")
            results["valid"] = False
            return results
        
        # Validate each sample
        sample_errors = []
        required_sample_fields = [
            "id", "prompt", "category", "difficulty", 
            "correct_answer", "reasoning_trace", "ddm_trajectory"
        ]
        
        for i, sample in enumerate(self.dataset["samples"]):
            for field in required_sample_fields:
                if field not in sample:
                    sample_errors.append(f"Sample {i} missing field: {field}")
            
            # Validate answer format
            if "correct_answer" in sample:
                if sample["correct_answer"] not in ["Yes", "No"]:
                    sample_errors.append(
                        f"Sample {i} has invalid answer: {sample['correct_answer']}"
                    )
            
            # Validate reasoning trace
            if "reasoning_trace" in sample:
                trace = sample["reasoning_trace"]
                if "tokens" not in trace or not isinstance(trace["tokens"], list):
                    sample_errors.append(f"Sample {i} has invalid reasoning tokens")
        
        if sample_errors:
            results["errors"].extend(sample_errors[:10])  # Show first 10 errors
            if len(sample_errors) > 10:
                results["errors"].append(f"... and {len(sample_errors) - 10} more errors")
            results["valid"] = False
        
        self.validation_results["structure"] = results
        return results
    
    def analyze_balance(self) -> Dict:
        """Analyze answer balance and distributions."""
        results = {
            "total_samples": len(self.dataset["samples"]),
            "answer_distribution": Counter(),
            "category_distribution": Counter(),
            "difficulty_distribution": Counter(),
            "category_answer_balance": defaultdict(Counter),
            "difficulty_answer_balance": defaultdict(Counter)
        }
        
        for sample in self.dataset["samples"]:
            answer = sample.get("correct_answer", "Unknown")
            category = sample.get("category", "Unknown")
            difficulty = sample.get("difficulty", "Unknown")
            
            results["answer_distribution"][answer] += 1
            results["category_distribution"][category] += 1
            results["difficulty_distribution"][difficulty] += 1
            
            results["category_answer_balance"][category][answer] += 1
            results["difficulty_answer_balance"][difficulty][answer] += 1
        
        # Calculate ratios
        total = results["total_samples"]
        if total > 0:
            yes_count = results["answer_distribution"].get("Yes", 0)
            no_count = results["answer_distribution"].get("No", 0)
            results["yes_ratio"] = yes_count / total
            results["no_ratio"] = no_count / total
            results["balance_score"] = 1 - abs(0.5 - results["yes_ratio"]) * 2
        
        self.validation_results["balance"] = results
        return results
    
    def analyze_reasoning_quality(self) -> Dict:
        """Analyze reasoning trace quality."""
        results = {
            "avg_tokens_per_sample": 0,
            "min_tokens": float('inf'),
            "max_tokens": 0,
            "empty_traces": 0,
            "token_length_distribution": [],
            "avg_evidence_scores": [],
            "reasoning_patterns": Counter()
        }
        
        token_counts = []
        
        for sample in self.dataset["samples"]:
            if "reasoning_trace" not in sample:
                results["empty_traces"] += 1
                continue
            
            trace = sample["reasoning_trace"]
            tokens = trace.get("tokens", [])
            num_tokens = len(tokens)
            
            token_counts.append(num_tokens)
            results["min_tokens"] = min(results["min_tokens"], num_tokens)
            results["max_tokens"] = max(results["max_tokens"], num_tokens)
            
            # Analyze token patterns
            for token in tokens:
                token_lower = token.lower()
                if "therefore" in token_lower or "thus" in token_lower:
                    results["reasoning_patterns"]["conclusion"] += 1
                elif "let me" in token_lower or "i need to" in token_lower:
                    results["reasoning_patterns"]["setup"] += 1
                elif "calculate" in token_lower or "compute" in token_lower:
                    results["reasoning_patterns"]["calculation"] += 1
        
        if token_counts:
            results["avg_tokens_per_sample"] = np.mean(token_counts)
            results["token_length_distribution"] = token_counts
        
        self.validation_results["reasoning"] = results
        return results
    
    def analyze_ddm_trajectories(self) -> Dict:
        """Analyze DDM trajectory characteristics."""
        results = {
            "avg_decision_time": 0,
            "threshold_reached_ratio": 0,
            "avg_final_evidence": 0,
            "evidence_distributions": {
                "final_evidence": [],
                "max_evidence": [],
                "evidence_variance": []
            },
            "decision_times": []
        }
        
        threshold_reached = 0
        decision_times = []
        final_evidences = []
        
        for sample in self.dataset["samples"]:
            if "ddm_trajectory" not in sample:
                continue
            
            trajectory = sample["ddm_trajectory"]
            
            # Decision time
            if "decision_time" in trajectory:
                decision_times.append(trajectory["decision_time"])
            
            # Threshold reached
            if trajectory.get("reached_threshold", False):
                threshold_reached += 1
            
            # Evidence analysis
            if "cumulative_evidence" in trajectory:
                evidence = trajectory["cumulative_evidence"]
                if evidence:
                    final_evidences.append(evidence[-1])
                    results["evidence_distributions"]["final_evidence"].append(evidence[-1])
                    results["evidence_distributions"]["max_evidence"].append(max(abs(e) for e in evidence))
                    if len(evidence) > 1:
                        results["evidence_distributions"]["evidence_variance"].append(np.var(evidence))
        
        # Calculate averages
        total_samples = len(self.dataset["samples"])
        if decision_times:
            results["avg_decision_time"] = np.mean(decision_times)
            results["decision_times"] = decision_times
        
        if total_samples > 0:
            results["threshold_reached_ratio"] = threshold_reached / total_samples
        
        if final_evidences:
            results["avg_final_evidence"] = np.mean(np.abs(final_evidences))
        
        self.validation_results["ddm"] = results
        return results
    
    def check_data_quality(self) -> Dict:
        """Check for data quality issues."""
        issues = {
            "duplicate_prompts": [],
            "mismatched_answers": [],
            "invalid_difficulties": [],
            "suspicious_patterns": [],
            "encoding_errors": []
        }
        
        seen_prompts = {}
        valid_difficulties = {"easy", "medium", "hard"}
        
        for i, sample in enumerate(self.dataset["samples"]):
            sample_id = sample.get("id", f"sample_{i}")
            
            # Check for duplicates
            prompt = sample.get("prompt", "")
            if prompt in seen_prompts:
                issues["duplicate_prompts"].append({
                    "samples": [seen_prompts[prompt], sample_id],
                    "prompt": prompt
                })
            else:
                seen_prompts[prompt] = sample_id
            
            # Check answer consistency
            correct_answer = sample.get("correct_answer")
            model_answer = sample.get("model_answer")
            if model_answer and correct_answer != model_answer:
                if sample.get("is_correct", True):  # Flag if marked correct but answers differ
                    issues["mismatched_answers"].append({
                        "sample": sample_id,
                        "correct": correct_answer,
                        "model": model_answer
                    })
            
            # Check difficulty values
            difficulty = sample.get("difficulty")
            if difficulty and difficulty not in valid_difficulties:
                issues["invalid_difficulties"].append({
                    "sample": sample_id,
                    "difficulty": difficulty
                })
            
            # Check for suspicious patterns
            if "reasoning_trace" in sample:
                tokens = sample["reasoning_trace"].get("tokens", [])
                # All tokens identical
                if len(tokens) > 1 and len(set(tokens)) == 1:
                    issues["suspicious_patterns"].append({
                        "sample": sample_id,
                        "issue": "All tokens identical"
                    })
                # Very short reasoning for complex problems
                if "hard" in str(difficulty) and len(tokens) < 3:
                    issues["suspicious_patterns"].append({
                        "sample": sample_id,
                        "issue": "Too short for hard problem"
                    })
        
        # Trim lists if too long
        for key in issues:
            if len(issues[key]) > 10:
                issues[key] = issues[key][:10]
                issues[f"{key}_count"] = len(issues[key])
        
        self.validation_results["quality"] = issues
        return issues
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate validation report."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("DDM DATASET VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Dataset: {self.dataset_path}")
        report_lines.append(f"Total Samples: {len(self.dataset['samples'])}")
        report_lines.append("")
        
        # Structure validation
        if "structure" in self.validation_results:
            struct = self.validation_results["structure"]
            report_lines.append("STRUCTURE VALIDATION")
            report_lines.append("-" * 30)
            report_lines.append(f"Valid: {struct['valid']}")
            if struct["errors"]:
                report_lines.append("Errors:")
                for error in struct["errors"]:
                    report_lines.append(f"  - {error}")
            if struct["warnings"]:
                report_lines.append("Warnings:")
                for warning in struct["warnings"]:
                    report_lines.append(f"  - {warning}")
            report_lines.append("")
        
        # Balance analysis
        if "balance" in self.validation_results:
            balance = self.validation_results["balance"]
            report_lines.append("ANSWER BALANCE")
            report_lines.append("-" * 30)
            report_lines.append(f"Yes: {balance['answer_distribution']['Yes']} ({balance.get('yes_ratio', 0):.1%})")
            report_lines.append(f"No: {balance['answer_distribution']['No']} ({balance.get('no_ratio', 0):.1%})")
            report_lines.append(f"Balance Score: {balance.get('balance_score', 0):.2f} (1.0 = perfect)")
            report_lines.append("")
            
            report_lines.append("Category Distribution:")
            for category, count in balance["category_distribution"].most_common():
                report_lines.append(f"  {category}: {count}")
            report_lines.append("")
        
        # Reasoning quality
        if "reasoning" in self.validation_results:
            reasoning = self.validation_results["reasoning"]
            report_lines.append("REASONING QUALITY")
            report_lines.append("-" * 30)
            report_lines.append(f"Avg tokens per sample: {reasoning['avg_tokens_per_sample']:.1f}")
            report_lines.append(f"Token range: {reasoning['min_tokens']} - {reasoning['max_tokens']}")
            report_lines.append(f"Empty traces: {reasoning['empty_traces']}")
            report_lines.append("")
        
        # DDM trajectories
        if "ddm" in self.validation_results:
            ddm = self.validation_results["ddm"]
            report_lines.append("DDM TRAJECTORIES")
            report_lines.append("-" * 30)
            report_lines.append(f"Avg decision time: {ddm['avg_decision_time']:.2f}")
            report_lines.append(f"Threshold reached: {ddm['threshold_reached_ratio']:.1%}")
            report_lines.append(f"Avg final evidence: {ddm['avg_final_evidence']:.2f}")
            report_lines.append("")
        
        # Quality issues
        if "quality" in self.validation_results:
            quality = self.validation_results["quality"]
            report_lines.append("QUALITY ISSUES")
            report_lines.append("-" * 30)
            
            issue_counts = {
                "duplicate_prompts": "Duplicate prompts",
                "mismatched_answers": "Mismatched answers",
                "invalid_difficulties": "Invalid difficulties",
                "suspicious_patterns": "Suspicious patterns"
            }
            
            for key, label in issue_counts.items():
                count = len(quality[key])
                if f"{key}_count" in quality:
                    count = quality[f"{key}_count"]
                if count > 0:
                    report_lines.append(f"{label}: {count}")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        
        return report
    
    def plot_statistics(self, output_dir: Optional[str] = None):
        """Generate statistical plots."""
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        else:
            output_path = Path(".")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Answer distribution pie chart
        if "balance" in self.validation_results:
            balance = self.validation_results["balance"]
            if balance["answer_distribution"]:
                plt.figure(figsize=(8, 6))
                plt.pie(
                    balance["answer_distribution"].values(),
                    labels=balance["answer_distribution"].keys(),
                    autopct='%1.1f%%',
                    colors=['#2ecc71', '#e74c3c']
                )
                plt.title("Answer Distribution")
                plt.savefig(output_path / "answer_distribution.png")
                plt.close()
        
        # 2. Token length distribution
        if "reasoning" in self.validation_results:
            reasoning = self.validation_results["reasoning"]
            if reasoning["token_length_distribution"]:
                plt.figure(figsize=(10, 6))
                plt.hist(reasoning["token_length_distribution"], bins=20, alpha=0.7, color='#3498db')
                plt.axvline(
                    reasoning["avg_tokens_per_sample"],
                    color='red',
                    linestyle='dashed',
                    linewidth=2,
                    label=f'Mean: {reasoning["avg_tokens_per_sample"]:.1f}'
                )
                plt.xlabel("Number of Reasoning Tokens")
                plt.ylabel("Frequency")
                plt.title("Distribution of Reasoning Token Counts")
                plt.legend()
                plt.savefig(output_path / "token_distribution.png")
                plt.close()
        
        # 3. Evidence trajectory plot
        if "ddm" in self.validation_results:
            ddm = self.validation_results["ddm"]
            if ddm["evidence_distributions"]["final_evidence"]:
                plt.figure(figsize=(10, 6))
                plt.hist(
                    ddm["evidence_distributions"]["final_evidence"],
                    bins=30,
                    alpha=0.7,
                    color='#9b59b6'
                )
                plt.xlabel("Final Evidence Value")
                plt.ylabel("Frequency")
                plt.title("Distribution of Final Evidence Values")
                plt.savefig(output_path / "evidence_distribution.png")
                plt.close()
        
        print(f"Plots saved to: {output_path}")
    
    def validate_all(self, generate_plots: bool = True) -> Dict:
        """Run all validation checks."""
        print("Running dataset validation...")
        
        # Run all validations
        self.validate_structure()
        self.analyze_balance()
        self.analyze_reasoning_quality()
        self.analyze_ddm_trajectories()
        self.check_data_quality()
        
        # Generate report
        report = self.generate_report()
        print("\n" + report)
        
        # Generate plots if requested
        if generate_plots:
            try:
                self.plot_statistics()
            except ImportError:
                print("Note: matplotlib not available for plotting")
        
        return self.validation_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate DDM dataset quality and statistics"
    )
    
    parser.add_argument(
        "dataset",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="validation_results",
        help="Output directory for plots and report"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate text report"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run validation
    validator = DDMDatasetValidator(args.dataset)
    results = validator.validate_all(generate_plots=not args.no_plots)
    
    # Save report
    report_path = output_dir / "validation_report.txt"
    validator.generate_report(str(report_path))
    
    # Save detailed results
    if not args.report_only:
        results_path = output_dir / "validation_results.json"
        with open(results_path, 'w') as f:
            # Convert Counter objects to dicts for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, Counter):
                            serializable_results[key][k] = dict(v)
                        elif isinstance(v, defaultdict):
                            serializable_results[key][k] = {kk: dict(vv) if isinstance(vv, Counter) else vv 
                                                           for kk, vv in v.items()}
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()