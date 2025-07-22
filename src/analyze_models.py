#!/usr/bin/env python3
"""
Model Comparison Analysis for Binary Reasoning Math Models

Analyzes and compares performance of all trained models:
- Logistic Regression
- Neural Network
- MC Dropout Bayesian
- Hierarchical Bayesian
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


def load_model_results():
    """Load all model results from JSON files."""
    models_dir = Path("models")
    
    results = {
        "Logistic Regression": json.load(open(models_dir / "binaryReasoningMath_model_results.json")),
        "Neural Network": json.load(open(models_dir / "binaryReasoningMath_neural_results.json")),
        "MC Dropout Bayesian": json.load(open(models_dir / "bayesian_mc_dropout_results.json")),
    }
    
    # Add hierarchical results from training output
    results["Hierarchical Bayesian"] = {
        "test_accuracy": 0.4467,
        "test_f1_score": 0.4469,
        "mean_uncertainty": 0.6923,
        "mean_epistemic": 0.0674,
        "mean_aleatoric": 0.6249,
        "category_performance": {
            "comparison": {"accuracy": 0.591, "epistemic": 0.0549, "aleatoric": 0.6373},
            "math_verification": {"accuracy": 0.471, "epistemic": 0.0492, "aleatoric": 0.6436},
            "number_property": {"accuracy": 0.558, "epistemic": 0.0698, "aleatoric": 0.6231},
            "pattern_recognition": {"accuracy": 0.450, "epistemic": 0.0524, "aleatoric": 0.6404}
        }
    }
    
    return results


def create_comparison_table(results):
    """Create a comparison table of all models."""
    data = []
    
    for model_name, result in results.items():
        row = {
            "Model": model_name,
            "Test Accuracy": f"{result['test_accuracy']*100:.1f}%",
            "F1 Score": f"{result['test_f1_score']:.3f}",
            "Val Accuracy": f"{result.get('best_val_accuracy', 0)*100:.1f}%" if result.get('best_val_accuracy') else "N/A",
            "Uncertainty": f"{result.get('mean_uncertainty', 0):.3f}" if 'mean_uncertainty' in result else "N/A",
            "ECE": f"{result.get('expected_calibration_error', 0):.3f}" if 'expected_calibration_error' in result else "N/A"
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def plot_model_comparison(results):
    """Create visualization comparing all models."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy Comparison
    ax = axes[0, 0]
    models = list(results.keys())
    accuracies = [results[m]['test_accuracy'] * 100 for m in models]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(models, accuracies, color=colors)
    ax.axhline(y=99.7, color='black', linestyle='--', alpha=0.5, label='Dataset Generation Accuracy')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    ax.legend()
    
    # 2. Uncertainty vs Accuracy (for Bayesian models)
    ax = axes[0, 1]
    bayesian_models = ["MC Dropout Bayesian", "Hierarchical Bayesian"]
    bay_acc = []
    bay_unc = []
    
    for model in bayesian_models:
        if model in results and 'mean_uncertainty' in results[model]:
            bay_acc.append(results[model]['test_accuracy'] * 100)
            bay_unc.append(results[model]['mean_uncertainty'])
    
    ax.scatter(bay_unc, bay_acc, s=200, alpha=0.7)
    for i, model in enumerate(bayesian_models):
        if i < len(bay_acc):
            ax.annotate(model, (bay_unc[i], bay_acc[i]), 
                       xytext=(10, 5), textcoords='offset points')
    
    ax.set_xlabel('Mean Uncertainty')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Uncertainty vs Accuracy Trade-off')
    ax.grid(True, alpha=0.3)
    
    # 3. F1 Score Comparison
    ax = axes[1, 0]
    f1_scores = [results[m]['test_f1_score'] for m in models]
    bars = ax.bar(models, f1_scores, color=colors)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison')
    ax.set_ylim(0, 1.1)
    
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom')
    
    # 4. Category Performance Heatmap (Hierarchical Model)
    ax = axes[1, 1]
    if 'category_performance' in results["Hierarchical Bayesian"]:
        cat_perf = results["Hierarchical Bayesian"]['category_performance']
        categories = list(cat_perf.keys())
        metrics = ['accuracy', 'epistemic', 'aleatoric']
        
        heatmap_data = []
        for cat in categories:
            row = [cat_perf[cat][m] for m in metrics]
            heatmap_data.append(row)
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(['Accuracy', 'Epistemic Unc.', 'Aleatoric Unc.'])
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        ax.set_title('Hierarchical Model: Category Performance')
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.3f}',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('models/model_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_timeline():
    """Create a simple timeline showing model progression."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = [
        ("Logistic Regression", 98.0, "Simple, interpretable"),
        ("Neural Network", 78.7, "Complex patterns"),
        ("MC Dropout Bayesian", 82.7, "+ Uncertainty"),
        ("Hierarchical Bayesian", 44.7, "+ Category-specific")
    ]
    
    x = range(len(models))
    accuracies = [m[1] for m in models]
    
    # Create bar chart
    bars = ax.bar(x, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add model names and descriptions
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m[0]}\n{m[2]}" for m in models], ha='center')
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Model Development Timeline: Accuracy vs Complexity Trade-off')
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='Random Baseline')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('models/model_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main analysis function."""
    print("Loading model results...")
    results = load_model_results()
    
    print("\nModel Comparison Table:")
    df = create_comparison_table(results)
    print(df.to_string(index=False))
    
    # Save table to file
    df.to_csv('models/model_comparison_table.csv', index=False)
    
    print("\nGenerating visualizations...")
    plot_model_comparison(results)
    plot_model_timeline()
    
    print("\nAnalysis Summary:")
    print("================")
    print(f"Best Accuracy: Logistic Regression - {results['Logistic Regression']['test_accuracy']*100:.1f}%")
    print(f"Best with Uncertainty: MC Dropout - {results['MC Dropout Bayesian']['test_accuracy']*100:.1f}%")
    print(f"Most Complex: Hierarchical Bayesian - {results['Hierarchical Bayesian']['test_accuracy']*100:.1f}%")
    
    # Calculate average epistemic vs aleatoric for hierarchical model
    hier_cat = results["Hierarchical Bayesian"]['category_performance']
    avg_epistemic = np.mean([cat['epistemic'] for cat in hier_cat.values()])
    avg_aleatoric = np.mean([cat['aleatoric'] for cat in hier_cat.values()])
    
    print(f"\nUncertainty Breakdown (Hierarchical):")
    print(f"  Epistemic (model): {avg_epistemic:.3f} ({avg_epistemic/(avg_epistemic+avg_aleatoric)*100:.1f}%)")
    print(f"  Aleatoric (data): {avg_aleatoric:.3f} ({avg_aleatoric/(avg_epistemic+avg_aleatoric)*100:.1f}%)")
    
    print("\nVisualizations saved to models/ directory")


if __name__ == "__main__":
    main()