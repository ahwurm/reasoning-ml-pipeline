#!/usr/bin/env python3
"""
Training pipeline for Binary Reasoning Math Model
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.binary_reasoning_math_model import BinaryReasoningMathModel


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


def train_neural_model(model, train_loader, val_loader, config, device):
    """Train neural network model."""
    model.model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.model.parameters(),
        lr=config.get('learning_rate', 0.001)
    )
    
    # Learning rate scheduler
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
            outputs = model.model(input_ids, attention_mask)
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
        
        # Validation phase
        model.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'\nEpoch {epoch+1}/{config.get("epochs", 50)}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            model.save(os.path.join('models', 'binaryReasoningMath_model_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
    
    return best_val_acc


def train_sklearn_model(model, X_train, y_train, X_val, y_val):
    """Train sklearn model."""
    model.model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred = model.model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f'Validation Accuracy: {val_acc:.4f}')
    return val_acc


def evaluate_model(model, test_samples, device=None):
    """Evaluate model on test set."""
    # Get predictions
    predictions = model.predict(test_samples)
    
    # Get true labels
    true_labels = []
    for sample in test_samples:
        label = 1 if sample['correct_answer'].lower() == 'yes' else 0
        true_labels.append(label)
    
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    print('\n=== Test Set Evaluation ===')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('\nClassification Report:')
    print(classification_report(true_labels, predictions, target_names=['No', 'Yes']))
    print('\nConfusion Matrix:')
    print(confusion_matrix(true_labels, predictions))
    
    return accuracy, f1


def main():
    parser = argparse.ArgumentParser(description='Train Binary Reasoning Math Model')
    parser.add_argument('--config', type=str, default='configs/binary_reasoning_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/api_dataset_incremental.json',
                        help='Path to dataset')
    parser.add_argument('--model-type', type=str, default='neural',
                        choices=['neural', 'random_forest', 'logistic'],
                        help='Type of model to train')
    parser.add_argument('--output', type=str, default='models/binaryReasoningMath_model.pth',
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
                'dropout': 0.3
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
    model = BinaryReasoningMathModel(
        model_type=args.model_type,
        **config.get('model', {})
    )
    
    # Build model
    model.build(train_samples)
    
    # Train model
    print(f'Training {args.model_type} model...')
    
    if args.model_type == 'neural':
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
        best_val_acc = train_neural_model(
            model, train_loader, val_loader, 
            config['training'], device
        )
        
    else:
        # Prepare features for sklearn models
        X_train = model.prepare_sklearn_features(train_samples)
        y_train = np.array([1 if s['correct_answer'].lower() == 'yes' else 0 for s in train_samples])
        
        X_val = model.prepare_sklearn_features(val_samples)
        y_val = np.array([1 if s['correct_answer'].lower() == 'yes' else 0 for s in val_samples])
        
        # Train sklearn model
        best_val_acc = train_sklearn_model(model, X_train, y_train, X_val, y_val)
    
    # Save final model
    print(f'\nSaving model to {args.output}...')
    model.save(args.output)
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    test_accuracy, test_f1 = evaluate_model(model, test_samples, device)
    
    # Save results
    results = {
        'model_type': args.model_type,
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_accuracy),
        'test_f1_score': float(test_f1),
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    results_path = args.output.replace('.pth', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nTraining complete! Results saved to {results_path}')
    print(f'Model saved to {args.output}')


if __name__ == '__main__':
    main()