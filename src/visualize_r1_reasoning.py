#!/usr/bin/env python3
"""
Simple visualization of R1 reasoning patterns.

Uses matplotlib for basic plots - no complex dependencies.
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


class R1Visualizer:
    """Create simple visualizations of R1 reasoning patterns."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
    
    def load_analysis(self, analysis_file: str) -> Dict:
        """Load analysis results from JSON file."""
        with open(analysis_file, 'r') as f:
            return json.load(f)
    
    def plot_token_distribution(self, analysis: Dict, save_path: Optional[str] = None):
        """Plot distribution of tokens per response."""
        response_stats = analysis.get("responses", [])
        if not response_stats:
            print("No response statistics found")
            return
        
        token_counts = [r["num_tokens"] for r in response_stats]
        
        plt.figure(figsize=(10, 6))
        plt.hist(token_counts, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Reasoning Tokens')
        plt.ylabel('Number of Responses')
        plt.title('Distribution of Reasoning Tokens per Response')
        
        # Add statistics
        avg_tokens = analysis["summary"]["avg_tokens_per_response"]
        plt.axvline(avg_tokens, color='red', linestyle='dashed', linewidth=2, 
                   label=f'Average: {avg_tokens:.1f}')
        plt.legend()
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved token distribution plot to {save_path}")
        else:
            save_path = self.output_dir / "token_distribution.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved token distribution plot to {save_path}")
        
        plt.close()
    
    def plot_token_types(self, analysis: Dict, save_path: Optional[str] = None):
        """Plot distribution of token types."""
        token_types = analysis["summary"].get("token_type_distribution", {})
        if not token_types:
            print("No token type data found")
            return
        
        # Sort by count
        types = list(token_types.keys())
        counts = list(token_types.values())
        
        # Create color map
        colors = plt.cm.Set3(range(len(types)))
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(types, counts, color=colors, edgecolor='black', alpha=0.8)
        
        plt.xlabel('Token Type')
        plt.ylabel('Count')
        plt.title('Distribution of Reasoning Token Types')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved token types plot to {save_path}")
        else:
            save_path = self.output_dir / "token_types.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved token types plot to {save_path}")
        
        plt.close()
    
    def plot_reasoning_flow(self, response: Dict, save_path: Optional[str] = None):
        """Visualize the flow of reasoning for a single response."""
        tokens = response.get("reasoning_tokens", [])
        if not tokens:
            print("No reasoning tokens in response")
            return
        
        # Limit tokens for readability
        max_tokens = 15
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            truncated = True
        else:
            truncated = False
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(tokens) * 0.8)))
        
        # Create boxes for each token
        y_positions = list(range(len(tokens)))
        box_height = 0.7
        box_width = 10
        
        for i, (y_pos, token) in enumerate(zip(y_positions, tokens)):
            # Truncate long tokens
            display_token = token[:80] + "..." if len(token) > 80 else token
            
            # Determine color based on token content
            if any(word in token.lower() for word in ["therefore", "thus", "so", "conclusion"]):
                color = 'lightgreen'
            elif any(word in token.lower() for word in ["but", "however", "although"]):
                color = 'lightcoral'
            elif any(word in token.lower() for word in ["maybe", "perhaps", "might"]):
                color = 'lightyellow'
            else:
                color = 'lightblue'
            
            # Create box
            rect = mpatches.FancyBboxPatch(
                (0, y_pos - box_height/2), box_width, box_height,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
            
            # Add text
            ax.text(box_width/2, y_pos, display_token,
                   ha='center', va='center', wrap=True, fontsize=9)
            
            # Add arrow to next token
            if i < len(tokens) - 1:
                ax.arrow(box_width/2, y_pos - box_height/2 - 0.05,
                        0, -0.4,
                        head_width=0.3, head_length=0.1,
                        fc='gray', ec='gray')
        
        # Set limits and labels
        ax.set_xlim(-0.5, box_width + 0.5)
        ax.set_ylim(-1, len(tokens))
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        prompt = response.get("prompt", "Unknown prompt")
        title = f"Reasoning Flow: {prompt[:60]}..."
        if truncated:
            title += f" (showing first {max_tokens} tokens)"
        plt.title(title, fontsize=12, pad=20)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightgreen', label='Conclusion'),
            mpatches.Patch(color='lightcoral', label='Contradiction'),
            mpatches.Patch(color='lightyellow', label='Uncertainty'),
            mpatches.Patch(color='lightblue', label='Statement')
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved reasoning flow to {save_path}")
        else:
            prompt_id = response.get("prompt_id", "unknown")
            save_path = self.output_dir / f"reasoning_flow_{prompt_id}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved reasoning flow to {save_path}")
        
        plt.close()
    
    def plot_pattern_frequency(self, analysis: Dict, save_path: Optional[str] = None):
        """Plot most common starting patterns."""
        top_patterns = analysis["summary"].get("top_patterns", [])[:15]
        if not top_patterns:
            print("No pattern data found")
            return
        
        patterns = [p[0] for p in top_patterns]
        counts = [p[1] for p in top_patterns]
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(patterns, counts, color='steelblue', edgecolor='black', alpha=0.8)
        
        plt.xlabel('Frequency')
        plt.ylabel('Pattern')
        plt.title('Most Common Token Starting Patterns')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}',
                    ha='left', va='center', fontweight='bold')
        
        plt.gca().invert_yaxis()  # Highest counts at top
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved pattern frequency plot to {save_path}")
        else:
            save_path = self.output_dir / "pattern_frequency.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved pattern frequency plot to {save_path}")
        
        plt.close()
    
    def create_summary_report(self, analysis: Dict, responses: List[Dict]):
        """Create a text summary report."""
        report_path = self.output_dir / "r1_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("R1 REASONING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            summary = analysis["summary"]
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total responses analyzed: {summary['total_responses']}\n")
            f.write(f"Total reasoning tokens: {summary['total_tokens']}\n")
            f.write(f"Average tokens per response: {summary['avg_tokens_per_response']:.1f}\n")
            f.write(f"Token range: {summary['min_tokens']} - {summary['max_tokens']}\n")
            f.write(f"Average token length: {summary['avg_token_length']:.1f} characters\n\n")
            
            # Token type distribution
            f.write("TOKEN TYPE DISTRIBUTION\n")
            f.write("-" * 20 + "\n")
            for token_type, count in summary['token_type_distribution'].items():
                percentage = (count / summary['total_tokens']) * 100
                f.write(f"{token_type:15} {count:6d} ({percentage:5.1f}%)\n")
            f.write("\n")
            
            # Most common patterns
            f.write("TOP 10 STARTING PATTERNS\n")
            f.write("-" * 20 + "\n")
            for pattern, count in summary['top_patterns'][:10]:
                f.write(f"'{pattern}': {count} occurrences\n")
            f.write("\n")
            
            # Example reasoning chains
            f.write("EXAMPLE REASONING CHAINS\n")
            f.write("-" * 20 + "\n")
            
            # Show up to 3 examples
            example_responses = [r for r in analysis["responses"] if r["num_tokens"] >= 3][:3]
            
            for i, resp_stat in enumerate(example_responses):
                f.write(f"\nExample {i+1}:\n")
                f.write(f"Prompt: {resp_stat['prompt_preview']}\n")
                f.write(f"Number of tokens: {resp_stat['num_tokens']}\n")
                f.write(f"Has calculation: {resp_stat['has_calculation']}\n")
                f.write(f"Has uncertainty: {resp_stat['has_uncertainty']}\n")
                
                # Find the actual response
                prompt_id = resp_stat['prompt_id']
                actual_response = next((r for r in responses if r.get('prompt_id') == prompt_id), None)
                
                if actual_response and 'reasoning_tokens' in actual_response:
                    f.write("Reasoning flow:\n")
                    for j, token in enumerate(actual_response['reasoning_tokens'][:5]):
                        f.write(f"  {j+1}. {token[:100]}{'...' if len(token) > 100 else ''}\n")
                    if len(actual_response['reasoning_tokens']) > 5:
                        f.write(f"  ... ({len(actual_response['reasoning_tokens']) - 5} more tokens)\n")
        
        print(f"Saved analysis report to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize R1 reasoning patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "analysis_file",
        nargs="?",
        default="data/r1_tokens_analysis.json",
        help="Path to analysis JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--responses-dir",
        default="data/r1_responses",
        help="Directory containing original responses (for flow diagrams)"
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        choices=["distribution", "types", "patterns", "flow", "all"],
        default=["all"],
        help="Types of plots to generate"
    )
    parser.add_argument(
        "--example-flows",
        type=int,
        default=3,
        help="Number of example flow diagrams to create"
    )
    
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return
    
    # Check if analysis file exists
    if not os.path.exists(args.analysis_file):
        print(f"Error: Analysis file not found: {args.analysis_file}")
        print("Run process_r1_tokens.py first to generate analysis")
        return
    
    # Initialize visualizer
    visualizer = R1Visualizer(output_dir=args.output_dir)
    
    # Load analysis
    print(f"Loading analysis from {args.analysis_file}")
    analysis = visualizer.load_analysis(args.analysis_file)
    
    # Generate plots
    plot_all = "all" in args.plot_types
    
    if plot_all or "distribution" in args.plot_types:
        print("\nGenerating token distribution plot...")
        visualizer.plot_token_distribution(analysis)
    
    if plot_all or "types" in args.plot_types:
        print("\nGenerating token types plot...")
        visualizer.plot_token_types(analysis)
    
    if plot_all or "patterns" in args.plot_types:
        print("\nGenerating pattern frequency plot...")
        visualizer.plot_pattern_frequency(analysis)
    
    if plot_all or "flow" in args.plot_types:
        print("\nGenerating reasoning flow diagrams...")
        
        # Load original responses for flow diagrams
        responses = []
        responses_dir = Path(args.responses_dir)
        if responses_dir.exists():
            for filepath in sorted(responses_dir.glob("prompt_*.json"))[:args.example_flows]:
                try:
                    with open(filepath, 'r') as f:
                        responses.append(json.load(f))
                except Exception as e:
                    print(f"  Error loading {filepath.name}: {e}")
        
        for response in responses:
            if "reasoning_tokens" in response and len(response["reasoning_tokens"]) > 0:
                visualizer.plot_reasoning_flow(response)
    
    # Generate text report
    print("\nGenerating analysis report...")
    responses = []
    responses_dir = Path(args.responses_dir)
    if responses_dir.exists():
        for filepath in sorted(responses_dir.glob("prompt_*.json")):
            try:
                with open(filepath, 'r') as f:
                    responses.append(json.load(f))
            except:
                pass
    
    visualizer.create_summary_report(analysis, responses)
    
    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()