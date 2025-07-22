#!/usr/bin/env python3
"""
Training pipeline for Bayesian Binary Reasoning Math Model
Includes uncertainty quantification and calibration metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
import pyro
from pyro.optim import Adam as PyroAdam
from pyro.infer import SVI, Trace_ELBO
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bayesian_binary_reasoning_model import BayesianBinaryReasoningModel
from src.models.hierarchical_bayesian_model import HierarchicalBayesianWrapper


class BinaryReasoningDataset(Dataset):
    """PyTorch dataset for binary reasoning data."""
    
    def __init__(self, samples, vocab, max_length=50):
        self.samples = samples
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Encode reasoning trace
        trace = sample.get('reasoning_trace', {}).get('tokens', [])
        encoded = []
        
        for token in trace:
            words = token.lower().split()
            for word in words:
                encoded.append(self.vocab.get(word, 1))  # 1 is <UNK>
        
        # Pad or truncate
        if len(encoded) < self.max_length:
            attention_mask = [1] * len(encoded) + [0] * (self.max_length - len(encoded))
            encoded = encoded + [0] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
            attention_mask = [1] * self.max_length
        
        # Convert answer to label (0 for No, 1 for Yes)
        label = 1 if sample['correct_answer'].lower() == 'yes' else 0
        
        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'prompt': sample['prompt']
        }


def load_dataset(data_path: str):
    """Load dataset from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data['samples']


def train_hierarchical_bayesian(model, train_samples, val_samples, config, device):
    """Train Hierarchical Bayesian model."""
    print("Training Hierarchical Bayesian Model...")
    
    # Build vocabulary
    reasoning_traces = [
        sample.get('reasoning_trace', {}).get('tokens', [])
        for sample in train_samples
    ]
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for trace in reasoning_traces:
        for token in trace:
            words = token.lower().split()
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
    
    # Prepare data with categories
    train_batches = []
    batch_size = config.get('batch_size', 32)
    for i in range(0, len(train_samples), batch_size):
        batch_samples = train_samples[i:i+batch_size]
        batch = model.prepare_batch(batch_samples, vocab)
        train_batches.append(batch)
    
    val_batch = model.prepare_batch(val_samples, vocab)
    
    # Setup SVI
    pyro.clear_param_store()
    optimizer = PyroAdam({"lr": config.get('learning_rate', 0.001)})
    svi = SVI(
        model.model, 
        model.inference.guide,
        optimizer,
        loss=Trace_ELBO()
    )
    
    best_val_acc = 0
    patience_counter = 0
    early_stopping_patience = config.get('early_stopping_patience', 10)
    
    for epoch in range(config.get('epochs', 30)):
        # Training
        model.model.train()
        train_loss = 0
        
        for batch in tqdm(train_batches, desc=f'Epoch {epoch+1}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model.inference.train_step(svi, batch)
            train_loss += loss
        
        avg_train_loss = train_loss / len(train_batches)
        
        # Validation
        val_batch_device = {k: v.to(device) for k, v in val_batch.items()}
        val_results = model.inference.predict_with_uncertainty(
            val_batch_device['input_ids'],
            val_batch_device['attention_mask'],
            val_batch_device['category_idx'],
            num_samples=50
        )
        
        val_predictions = val_results['predictions'].cpu().numpy()
        val_labels = val_batch['label'].numpy()
        val_acc = np.mean(val_predictions == val_labels) * 100
        
        # Category-specific results
        cat_results = model.inference.get_category_specific_predictions(val_samples, 50)
        
        print(f'\nEpoch {epoch+1}/{config.get("epochs", 30)}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Acc: {val_acc:.2f}%')
        print(f'  Category-specific accuracy:')
        for cat, res in cat_results.items():
            print(f'    {cat}: {res["accuracy"]*100:.1f}% (uncertainty: {res["mean_uncertainty"]:.3f})')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save model
            torch.save({
                'model_state': model.model.state_dict(),
                'guide_state': pyro.get_param_store().get_state(),
                'vocab': vocab,
                'config': config
            }, os.path.join('models', 'hierarchical_bayesian_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
    
    # Analyze hierarchical structure
    print("\nAnalyzing hierarchical parameters...")
    hierarchy_analysis = model.inference.analyze_hierarchical_parameters()
    
    return best_val_acc, hierarchy_analysis


def train_bayesian_model(model, train_loader, val_loader, config, device):
    """Train Bayesian model (BNN or MC Dropout)."""
    
    if model.model_type == "bnn":
        # Train using Pyro's SVI
        print("Training Bayesian Neural Network with SVI...")
        
        # Clear parameter store
        pyro.clear_param_store()
        
        # Setup SVI
        from pyro.infer import SVI, Trace_ELBO
        from pyro.optim import Adam
        
        optimizer = PyroAdam({"lr": config.get('learning_rate', 0.001)})
        
        def model_fn(input_ids, attention_mask):
            with pyro.plate("data", input_ids.shape[0]):
                logits = model.model(input_ids, attention_mask)
                return logits
        
        def guide_fn(input_ids, attention_mask):
            # AutoGuide will handle this
            pass
        
        # Use AutoGuide
        from pyro.infer.autoguide import AutoDiagonalNormal
        guide = AutoDiagonalNormal(model.model)
        model.guide = guide
        
        svi = SVI(model.model, guide, optimizer, loss=Trace_ELBO())
        
        best_val_acc = 0
        patience_counter = 0
        early_stopping_patience = config.get('early_stopping_patience', 10)
        
        for epoch in range(config.get('epochs', 50)):
            # Training phase
            model.model.train()
            train_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Custom loss function for classification
                def model_with_loss(input_ids, attention_mask):
                    logits = model.model(input_ids, attention_mask)
                    # Sample from categorical distribution
                    with pyro.plate("data", input_ids.shape[0]):
                        pyro.sample("obs", pyro.distributions.Categorical(logits=logits), obs=labels)
                    return logits
                
                loss = svi.step(input_ids, attention_mask)
                train_loss += loss
                
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            # Validation phase
            val_correct = 0
            val_total = 0
            val_samples = []
            
            with torch.no_grad():
                for batch in val_loader:
                    val_samples.extend([{
                        'prompt': p,
                        'reasoning_trace': {'tokens': model.samples[i].get('reasoning_trace', {}).get('tokens', [])}
                    } for i, p in enumerate(batch['prompt'])])
                    
                    labels = batch['label'].numpy()
                    val_total += len(labels)
            
            # Get predictions with uncertainty
            results = model.predict_with_uncertainty(val_samples, num_samples=20)
            val_predictions = results['predictions']
            val_correct = np.sum(val_predictions == labels)
            
            val_acc = 100. * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            
            print(f'\nEpoch {epoch+1}/{config.get("epochs", 50)}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model.save(os.path.join('models', 'bayesian_model_best.pth'))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
    
    elif model.model_type == "mc_dropout":
        # Train MC Dropout model like a regular neural network
        print("Training MC Dropout Neural Network...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        
        best_val_acc = 0
        patience_counter = 0
        early_stopping_patience = config.get('early_stopping_patience', 10)
        
        for epoch in range(config.get('epochs', 50)):
            # Training phase
            model.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model.model(input_ids, attention_mask, mc_samples=1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation with uncertainty
            model.model.train()  # Keep dropout active
            val_loss = 0
            val_predictions = []
            val_labels = []
            val_uncertainties = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Multiple forward passes for uncertainty
                    outputs = model.model(input_ids, attention_mask, mc_samples=10)
                    probs = F.softmax(outputs, dim=-1)
                    mean_probs = probs.mean(dim=0)
                    
                    _, predicted = torch.max(mean_probs, 1)
                    val_predictions.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    
                    # Calculate uncertainty as entropy
                    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
                    val_uncertainties.extend(entropy.cpu().numpy())
            
            val_acc = accuracy_score(val_labels, val_predictions) * 100
            avg_train_loss = train_loss / len(train_loader)
            
            print(f'\nEpoch {epoch+1}/{config.get("epochs", 50)}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {100.*train_correct/train_total:.2f}%')
            print(f'  Val Acc: {val_acc:.2f}%, Avg Uncertainty: {np.mean(val_uncertainties):.4f}')
            
            scheduler.step(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model.save(os.path.join('models', 'bayesian_model_best.pth'))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
    
    return best_val_acc


def train_gaussian_process(model, train_samples, val_samples):
    """Train Gaussian Process classifier."""
    print("Training Gaussian Process Classifier...")
    
    # Prepare features
    X_train = model.vectorizer.fit_transform([
        s['prompt'] + ' ' + ' '.join(s.get('reasoning_trace', {}).get('tokens', []))
        for s in train_samples
    ])
    y_train = np.array([1 if s['correct_answer'].lower() == 'yes' else 0 for s in train_samples])
    
    X_val = model.vectorizer.transform([
        s['prompt'] + ' ' + ' '.join(s.get('reasoning_trace', {}).get('tokens', []))
        for s in val_samples
    ])
    y_val = np.array([1 if s['correct_answer'].lower() == 'yes' else 0 for s in val_samples])
    
    # Train GP
    model.model.fit(X_train, y_train)
    
    # Evaluate
    val_pred = model.model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred) * 100
    
    print(f'Validation Accuracy: {val_acc:.2f}%')
    return val_acc


def evaluate_hierarchical_model(model, test_samples, vocab, device):
    """Evaluate hierarchical Bayesian model."""
    # Prepare test batch
    test_batch = model.prepare_batch(test_samples, vocab)
    test_batch = {k: v.to(device) for k, v in test_batch.items()}
    
    # Get predictions with uncertainty
    results = model.inference.predict_with_uncertainty(
        test_batch['input_ids'],
        test_batch['attention_mask'],
        test_batch['category_idx'],
        num_samples=100
    )
    
    predictions = results['predictions'].cpu().numpy()
    probabilities = results['probabilities'].cpu().numpy()
    uncertainties = results['uncertainty_total'].cpu().numpy()
    epistemic = results['uncertainty_epistemic'].cpu().numpy()
    aleatoric = results['uncertainty_aleatoric'].cpu().numpy()
    
    # True labels
    true_labels = test_batch['label'].cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    # Calculate calibration error
    positive_probs = probabilities[:, 1]
    from sklearn.calibration import calibration_curve
    
    print('\n=== Hierarchical Bayesian Model Evaluation ===')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Mean Total Uncertainty: {np.mean(uncertainties):.4f}')
    print(f'Mean Epistemic Uncertainty: {np.mean(epistemic):.4f}')
    print(f'Mean Aleatoric Uncertainty: {np.mean(aleatoric):.4f}')
    
    # Category-specific results
    cat_results = model.inference.get_category_specific_predictions(test_samples, 100)
    print('\nCategory-specific Performance:')
    for cat, res in cat_results.items():
        print(f'  {cat}:')
        print(f'    Accuracy: {res["accuracy"]*100:.1f}%')
        print(f'    Epistemic Uncertainty: {res["mean_epistemic"]:.4f}')
        print(f'    Aleatoric Uncertainty: {res["mean_aleatoric"]:.4f}')
    
    print('\nClassification Report:')
    print(classification_report(true_labels, predictions, target_names=['No', 'Yes']))
    
    # Plot hierarchical uncertainty analysis
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Uncertainty by category
    plt.subplot(1, 3, 1)
    categories = []
    for sample in test_samples:
        categories.append(sample.get('category', 'unknown'))
    
    unique_cats = list(set(categories))
    cat_uncertainties = {cat: [] for cat in unique_cats}
    
    for i, cat in enumerate(categories):
        cat_uncertainties[cat].append(uncertainties[i])
    
    plt.boxplot([cat_uncertainties[cat] for cat in unique_cats],
                labels=unique_cats)
    plt.xticks(rotation=45)
    plt.ylabel('Total Uncertainty')
    plt.title('Uncertainty Distribution by Category')
    
    # Plot 2: Epistemic vs Aleatoric
    plt.subplot(1, 3, 2)
    plt.scatter(epistemic, aleatoric, alpha=0.5, 
                c=['green' if p == t else 'red' for p, t in zip(predictions, true_labels)])
    plt.xlabel('Epistemic Uncertainty')
    plt.ylabel('Aleatoric Uncertainty')
    plt.title('Epistemic vs Aleatoric Uncertainty')
    
    # Plot 3: Calibration by category
    plt.subplot(1, 3, 3)
    for cat in unique_cats[:4]:  # Top 4 categories
        cat_mask = np.array([c == cat for c in categories])
        if cat_mask.sum() > 10:
            cat_probs = positive_probs[cat_mask]
            cat_labels = true_labels[cat_mask]
            try:
                fraction_pos, mean_pred = calibration_curve(cat_labels, cat_probs, n_bins=5)
                plt.plot(mean_pred, fraction_pos, 'o-', label=cat)
            except:
                pass
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration by Category')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/hierarchical_uncertainty_analysis.png', dpi=150)
    plt.close()
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'mean_uncertainty': np.mean(uncertainties),
        'mean_epistemic': np.mean(epistemic),
        'mean_aleatoric': np.mean(aleatoric),
        'predictions': predictions,
        'probabilities': probabilities,
        'uncertainties': uncertainties,
        'category_results': cat_results
    }


def evaluate_bayesian_model(model, test_samples, device=None):
    """Evaluate Bayesian model with uncertainty metrics."""
    # Get predictions with uncertainty
    results = model.predict_with_uncertainty(test_samples, num_samples=100)
    predictions = results['predictions']
    probabilities = results['probabilities']
    uncertainties = results['uncertainties']
    
    # Get true labels
    true_labels = np.array([
        1 if sample['correct_answer'].lower() == 'yes' else 0 
        for sample in test_samples
    ])
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    # Calculate calibration error
    positive_probs = probabilities[:, 1]  # Probability of class 1
    ece = model.calibration_error(positive_probs, true_labels)
    
    print('\n=== Bayesian Model Evaluation ===')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Expected Calibration Error: {ece:.4f}')
    print(f'Mean Uncertainty: {np.mean(uncertainties):.4f}')
    print('\nClassification Report:')
    print(classification_report(true_labels, predictions, target_names=['No', 'Yes']))
    print('\nConfusion Matrix:')
    print(confusion_matrix(true_labels, predictions))
    
    # Plot uncertainty calibration
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Uncertainty vs Accuracy
    plt.subplot(1, 3, 1)
    correct = predictions == true_labels
    plt.scatter(uncertainties[correct], np.ones(sum(correct)), alpha=0.5, label='Correct', s=10)
    plt.scatter(uncertainties[~correct], np.zeros(sum(~correct)), alpha=0.5, label='Incorrect', s=10)
    plt.xlabel('Uncertainty')
    plt.ylabel('Correctness')
    plt.title('Uncertainty vs Prediction Correctness')
    plt.legend()
    
    # Plot 2: Calibration plot
    plt.subplot(1, 3, 2)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (positive_probs > bin_lower) & (positive_probs <= bin_upper)
        if in_bin.sum() > 0:
            accuracies.append(true_labels[in_bin].mean())
            confidences.append(positive_probs[in_bin].mean())
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(confidences, accuracies, 'o-', label='Model calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Accuracy')
    plt.title('Calibration Plot')
    plt.legend()
    
    # Plot 3: Uncertainty distribution
    plt.subplot(1, 3, 3)
    plt.hist(uncertainties[correct], alpha=0.5, label='Correct', bins=20, density=True)
    plt.hist(uncertainties[~correct], alpha=0.5, label='Incorrect', bins=20, density=True)
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title('Uncertainty Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/uncertainty_analysis.png', dpi=150)
    plt.close()
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'ece': ece,
        'mean_uncertainty': np.mean(uncertainties),
        'predictions': predictions,
        'probabilities': probabilities,
        'uncertainties': uncertainties
    }


def main():
    parser = argparse.ArgumentParser(description='Train Bayesian Binary Reasoning Model')
    parser.add_argument('--config', type=str, default='configs/bayesian_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/api_dataset_incremental.json',
                        help='Path to dataset')
    parser.add_argument('--model-type', type=str, default='mc_dropout',
                        choices=['bnn', 'mc_dropout', 'gaussian_process', 'hierarchical'],
                        help='Type of Bayesian model to train')
    parser.add_argument('--output', type=str, default='models/bayesian_model.pth',
                        help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = {
            'model': {
                'embedding_dim': 128,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.5,  # Higher dropout for MC Dropout
                'prior_scale': 1.0
            },
            'training': {
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.001,
                'early_stopping_patience': 10
            },
            'data': {
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'max_sequence_length': 50
            }
        }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    print(f'Loading dataset from {args.data}...')
    samples = load_dataset(args.data)
    print(f'Loaded {len(samples)} samples')
    
    # Split dataset
    train_val_samples, test_samples = train_test_split(
        samples, 
        test_size=config['data']['test_split'],
        random_state=42,
        stratify=[1 if s['correct_answer'].lower() == 'yes' else 0 for s in samples]
    )
    
    train_samples, val_samples = train_test_split(
        train_val_samples,
        test_size=config['data']['val_split'] / (1 - config['data']['test_split']),
        random_state=42,
        stratify=[1 if s['correct_answer'].lower() == 'yes' else 0 for s in train_val_samples]
    )
    
    print(f'Dataset split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}')
    
    # Initialize model
    print(f'Initializing {args.model_type} model...')
    
    if args.model_type == 'hierarchical':
        # Count vocabulary size
        reasoning_traces = [
            sample.get('reasoning_trace', {}).get('tokens', [])
            for sample in train_samples
        ]
        vocab = {}
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1
        for trace in reasoning_traces:
            for token in trace:
                words = token.lower().split()
                for word in words:
                    if word not in vocab:
                        vocab[word] = len(vocab)
        
        model = HierarchicalBayesianWrapper(
            vocab_size=len(vocab),
            num_categories=4,
            **config.get('model', {})
        )
    else:
        model = BayesianBinaryReasoningModel(
            model_type=args.model_type,
            **config.get('model', {})
        )
        # Build model
        model.build(train_samples)
    
    # Train model
    print(f'Training {args.model_type} model...')
    
    if args.model_type == 'hierarchical':
        # Train hierarchical Bayesian model
        best_val_acc, hierarchy_analysis = train_hierarchical_bayesian(
            model, train_samples, val_samples,
            config['training'], device
        )
        
    elif args.model_type in ['bnn', 'mc_dropout']:
        # Create datasets
        train_dataset = BinaryReasoningDataset(
            train_samples, 
            model.vocab,
            max_length=config['data']['max_sequence_length']
        )
        val_dataset = BinaryReasoningDataset(
            val_samples,
            model.vocab,
            max_length=config['data']['max_sequence_length']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        # Train neural model
        best_val_acc = train_bayesian_model(
            model, train_loader, val_loader, 
            config['training'], device
        )
        
    else:  # gaussian_process
        best_val_acc = train_gaussian_process(model, train_samples, val_samples)
    
    # Save final model
    print(f'\nSaving model to {args.output}...')
    if args.model_type == 'hierarchical':
        # Save hierarchical model
        torch.save({
            'model_state': model.model.state_dict(),
            'guide_state': pyro.get_param_store().get_state() if pyro.get_param_store() else {},
            'vocab': vocab if 'vocab' in locals() else {},
            'config': config,
            'model_type': 'hierarchical'
        }, args.output)
    else:
        model.save(args.output)
    
    # Evaluate on test set with uncertainty analysis
    print('\nEvaluating on test set...')
    if args.model_type == 'hierarchical':
        # Special evaluation for hierarchical model
        test_results = evaluate_hierarchical_model(model, test_samples, vocab, device)
    else:
        test_results = evaluate_bayesian_model(model, test_samples, device)
    
    # Save results
    results = {
        'model_type': args.model_type,
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_results['accuracy']),
        'test_f1_score': float(test_results['f1_score']),
        'expected_calibration_error': float(test_results['ece']),
        'mean_uncertainty': float(test_results['mean_uncertainty']),
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    results_path = args.output.replace('.pth', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nTraining complete! Results saved to {results_path}')
    print(f'Model saved to {args.output}')
    print(f'Uncertainty analysis plots saved to models/uncertainty_analysis.png')


if __name__ == '__main__':
    main()