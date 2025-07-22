#!/usr/bin/env python3
"""
Process and analyze R1 reasoning tokens.

Features:
- Load and parse collected R1 responses
- Extract and clean reasoning tokens
- Basic statistical analysis
- Export to CSV/JSON for further analysis
"""
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class R1TokenProcessor:
    """Process and analyze R1 reasoning tokens."""
    
    def __init__(self, data_dir: str = "data/r1_responses"):
        """
        Initialize token processor.
        
        Args:
            data_dir: Directory containing R1 response files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
    
    def load_responses(self, pattern: str = "*.json") -> List[Dict]:
        """
        Load all response files matching pattern.
        
        Args:
            pattern: Glob pattern for files
            
        Returns:
            List of response dictionaries
        """
        responses = []
        files = sorted(self.data_dir.glob(pattern))
        
        print(f"Loading responses from {self.data_dir}")
        for filepath in files:
            if filepath.name == "collection_summary.json":
                continue  # Skip summary file
                
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    data['source_file'] = filepath.name
                    responses.append(data)
            except Exception as e:
                print(f"  Error loading {filepath.name}: {e}")
        
        print(f"Loaded {len(responses)} responses")
        return responses
    
    def clean_token(self, token: str) -> str:
        """
        Clean a reasoning token.
        
        Args:
            token: Raw token text
            
        Returns:
            Cleaned token
        """
        # Remove extra whitespace
        token = ' '.join(token.split())
        
        # Remove markdown formatting
        token = re.sub(r'\*+', '', token)  # Remove asterisks
        token = re.sub(r'`+', '', token)   # Remove backticks
        
        # Normalize quotes
        token = token.replace('"', '"').replace('"', '"')
        token = token.replace(''', "'").replace(''', "'")
        
        return token.strip()
    
    def extract_token_features(self, token: str) -> Dict[str, any]:
        """
        Extract features from a reasoning token.
        
        Args:
            token: Reasoning token
            
        Returns:
            Dictionary of features
        """
        features = {
            "length": len(token),
            "word_count": len(token.split()),
            "has_number": bool(re.search(r'\d', token)),
            "has_equation": bool(re.search(r'[=+\-*/]', token)),
            "has_question": token.strip().endswith('?'),
            "starts_with_so": token.lower().startswith(('so ', 'therefore', 'thus')),
            "starts_with_but": token.lower().startswith(('but ', 'however', 'although')),
            "is_conclusion": bool(re.search(r'(therefore|thus|so|in conclusion|finally)', token.lower())),
            "is_uncertainty": bool(re.search(r'(maybe|perhaps|might|could|possibly)', token.lower())),
        }
        
        # Token type classification
        if features["has_equation"] or re.search(r'\d+\s*[+\-*/]\s*\d+', token):
            features["type"] = "calculation"
        elif features["has_question"]:
            features["type"] = "question"
        elif features["is_conclusion"]:
            features["type"] = "conclusion"
        elif features["is_uncertainty"]:
            features["type"] = "uncertainty"
        elif features["starts_with_but"]:
            features["type"] = "contradiction"
        else:
            features["type"] = "statement"
        
        return features
    
    def analyze_responses(self, responses: List[Dict]) -> Dict[str, any]:
        """
        Analyze all responses and extract statistics.
        
        Args:
            responses: List of response dictionaries
            
        Returns:
            Analysis results
        """
        analysis = {
            "total_responses": len(responses),
            "total_tokens": 0,
            "tokens_per_response": [],
            "token_lengths": [],
            "token_types": Counter(),
            "common_patterns": Counter(),
            "response_stats": []
        }
        
        all_tokens = []
        
        for response in responses:
            if "error" in response:
                continue
                
            tokens = response.get("reasoning_tokens", [])
            prompt = response.get("prompt", "")
            final_answer = response.get("final_answer", "")
            
            # Clean tokens
            cleaned_tokens = [self.clean_token(t) for t in tokens if t.strip()]
            
            # Extract features for each token
            token_features = [self.extract_token_features(t) for t in cleaned_tokens]
            
            # Response-level stats
            response_stat = {
                "prompt_id": response.get("prompt_id", "unknown"),
                "prompt_preview": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "num_tokens": len(cleaned_tokens),
                "avg_token_length": sum(f["length"] for f in token_features) / len(token_features) if token_features else 0,
                "token_types": Counter(f["type"] for f in token_features),
                "final_answer_length": len(final_answer),
                "has_calculation": any(f["has_equation"] for f in token_features),
                "has_uncertainty": any(f["is_uncertainty"] for f in token_features),
            }
            
            analysis["response_stats"].append(response_stat)
            analysis["tokens_per_response"].append(len(cleaned_tokens))
            analysis["total_tokens"] += len(cleaned_tokens)
            
            # Aggregate token features
            for feature in token_features:
                analysis["token_lengths"].append(feature["length"])
                analysis["token_types"][feature["type"]] += 1
            
            # Pattern extraction
            for token in cleaned_tokens:
                # Extract common starting patterns
                words = token.split()
                if words:
                    first_word = words[0].lower()
                    analysis["common_patterns"][first_word] += 1
                    
                    if len(words) >= 2:
                        bigram = f"{words[0].lower()} {words[1].lower()}"
                        analysis["common_patterns"][bigram] += 1
            
            all_tokens.extend(cleaned_tokens)
        
        # Calculate summary statistics
        if analysis["tokens_per_response"]:
            analysis["avg_tokens_per_response"] = sum(analysis["tokens_per_response"]) / len(analysis["tokens_per_response"])
            analysis["min_tokens"] = min(analysis["tokens_per_response"])
            analysis["max_tokens"] = max(analysis["tokens_per_response"])
        
        if analysis["token_lengths"]:
            analysis["avg_token_length"] = sum(analysis["token_lengths"]) / len(analysis["token_lengths"])
            analysis["min_token_length"] = min(analysis["token_lengths"])
            analysis["max_token_length"] = max(analysis["token_lengths"])
        
        # Most common patterns (top 20)
        analysis["top_patterns"] = analysis["common_patterns"].most_common(20)
        
        # Store all tokens for export
        analysis["all_tokens"] = all_tokens
        
        return analysis
    
    def export_to_csv(self, analysis: Dict, output_file: str = "data/r1_tokens_analysis.csv"):
        """Export analysis results to CSV."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export response-level stats
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if analysis["response_stats"]:
                # Get all possible token types
                all_types = set()
                for stat in analysis["response_stats"]:
                    all_types.update(stat["token_types"].keys())
                all_types = sorted(all_types)
                
                # Define fieldnames
                fieldnames = [
                    "prompt_id", "prompt_preview", "num_tokens", 
                    "avg_token_length", "final_answer_length",
                    "has_calculation", "has_uncertainty"
                ] + [f"type_{t}" for t in all_types]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for stat in analysis["response_stats"]:
                    row = {
                        "prompt_id": stat["prompt_id"],
                        "prompt_preview": stat["prompt_preview"],
                        "num_tokens": stat["num_tokens"],
                        "avg_token_length": f"{stat['avg_token_length']:.1f}",
                        "final_answer_length": stat["final_answer_length"],
                        "has_calculation": stat["has_calculation"],
                        "has_uncertainty": stat["has_uncertainty"],
                    }
                    
                    # Add token type counts
                    for t in all_types:
                        row[f"type_{t}"] = stat["token_types"].get(t, 0)
                    
                    writer.writerow(row)
        
        print(f"Exported response analysis to {output_path}")
        
        # Export tokens to separate file
        tokens_file = output_path.parent / "r1_all_tokens.csv"
        with open(tokens_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["token_index", "token", "length", "word_count"])
            
            for i, token in enumerate(analysis.get("all_tokens", [])):
                writer.writerow([i, token, len(token), len(token.split())])
        
        print(f"Exported all tokens to {tokens_file}")
    
    def export_to_json(self, analysis: Dict, output_file: str = "data/r1_tokens_analysis.json"):
        """Export analysis results to JSON."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        export_data = {
            "analysis_date": datetime.now().isoformat(),
            "summary": {
                "total_responses": analysis["total_responses"],
                "total_tokens": analysis["total_tokens"],
                "avg_tokens_per_response": analysis.get("avg_tokens_per_response", 0),
                "min_tokens": analysis.get("min_tokens", 0),
                "max_tokens": analysis.get("max_tokens", 0),
                "avg_token_length": analysis.get("avg_token_length", 0),
                "token_type_distribution": dict(analysis["token_types"]),
                "top_patterns": analysis["top_patterns"]
            },
            "responses": analysis["response_stats"],
            "all_tokens": analysis["all_tokens"][:100]  # Sample of tokens
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported full analysis to {output_path}")
    
    def print_summary(self, analysis: Dict):
        """Print analysis summary to console."""
        print("\n=== R1 Token Analysis Summary ===")
        print(f"Total responses analyzed: {analysis['total_responses']}")
        print(f"Total reasoning tokens: {analysis['total_tokens']}")
        
        if analysis.get('avg_tokens_per_response'):
            print(f"\nTokens per response:")
            print(f"  Average: {analysis['avg_tokens_per_response']:.1f}")
            print(f"  Min: {analysis['min_tokens']}")
            print(f"  Max: {analysis['max_tokens']}")
        
        if analysis.get('avg_token_length'):
            print(f"\nToken lengths:")
            print(f"  Average: {analysis['avg_token_length']:.1f} characters")
            print(f"  Min: {analysis['min_token_length']} characters")
            print(f"  Max: {analysis['max_token_length']} characters")
        
        if analysis['token_types']:
            print(f"\nToken type distribution:")
            for token_type, count in analysis['token_types'].most_common():
                percentage = (count / analysis['total_tokens']) * 100
                print(f"  {token_type}: {count} ({percentage:.1f}%)")
        
        if analysis.get('top_patterns'):
            print(f"\nMost common patterns:")
            for pattern, count in analysis['top_patterns'][:10]:
                print(f"  '{pattern}': {count} occurrences")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process and analyze R1 reasoning tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        default="data/r1_responses",
        help="Directory containing R1 response files"
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="File pattern to match (default: *.json)"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Output format"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Don't print summary to console"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    try:
        processor = R1TokenProcessor(data_dir=args.data_dir)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Load responses
    responses = processor.load_responses(pattern=args.pattern)
    if not responses:
        print("No responses found to process")
        return
    
    # Analyze
    print("\nAnalyzing tokens...")
    analysis = processor.analyze_responses(responses)
    
    # Export results
    if args.format in ["csv", "both"]:
        csv_file = Path(args.output_dir) / "r1_tokens_analysis.csv"
        processor.export_to_csv(analysis, str(csv_file))
    
    if args.format in ["json", "both"]:
        json_file = Path(args.output_dir) / "r1_tokens_analysis.json"
        processor.export_to_json(analysis, str(json_file))
    
    # Print summary
    if not args.no_summary:
        processor.print_summary(analysis)


if __name__ == "__main__":
    main()