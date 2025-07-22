#!/usr/bin/env python3
"""
Train MC Dropout model on viral debates dataset with reasoning traces.
Focus on uncertainty quantification and consistency across variations.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import argparse
from tqdm import tqdm
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DebatesDataset(Dataset):
    """Dataset for viral debates with reasoning traces."""
    
    def __init__(self, questions: List[Dict], vectorizer: Optional[TfidfVectorizer] = None, 
                 scaler: Optional[StandardScaler] = None, fit_transform: bool = True):
        """Initialize dataset."""
        self.questions = questions
        self.vectorizer = vectorizer or TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.scaler = scaler or StandardScaler()
        
        # Extract features
        self.features, self.labels = self._extract_features(fit_transform)
        
    def _extract_features(self, fit_transform: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from questions and reasoning."""
        # Combine question and reasoning for feature extraction
        texts = []
        labels = []
        
        for q in self.questions:
            # Combine question and reasoning
            text = f"{q['question']} {q.get('reasoning', '')}"
            texts.append(text)
            
            # Convert answer to binary label
            answer = q.get('answer', 'unknown')
            if answer == 'yes':
                labels.append(1)
            elif answer == 'no':
                labels.append(0)
            else:
                # Skip unknown answers
                continue
        
        # Vectorize text
        if fit_transform:
            text_features = self.vectorizer.fit_transform(texts).toarray()
        else:
            text_features = self.vectorizer.transform(texts).toarray()
        
        # Add additional features
        additional_features = []
        for q in self.questions:
            if q.get('answer', 'unknown') in ['yes', 'no']:
                features = [
                    q.get('confidence', 0.5),
                    len(q.get('reasoning', '').split()),
                    1 if q['polarity'] == 'positive' else 0,
                    q['variation_num'] / 5.0,  # Normalize variation number
                    1 if q.get('is_qc', False) else 0
                ]
                additional_features.append(features)
        
        additional_features = np.array(additional_features)
        
        # Combine all features
        if fit_transform:
            additional_features = self.scaler.fit_transform(additional_features)
        else:
            additional_features = self.scaler.transform(additional_features)
        
        features = np.hstack([text_features, additional_features])
        labels = np.array(labels)
        
        return features, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])


class MCDropoutModel(nn.Module):
    """Neural network with MC Dropout for uncertainty estimation."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128], 
                 dropout_rate: float = 0.25):
        """Initialize model."""
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        return self.model(x)
    
    def predict_with_uncertainty(self, x, n_samples: int = 20):
        """Make predictions with uncertainty estimation."""
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred


def analyze_consistency(dataset: Dict, predictions: Dict) -> Dict:
    """Analyze consistency across question variations."""
    from collections import defaultdict
    
    consistency_results = defaultdict(lambda: {
        'positive': [],
        'negative': [],
        'variations': defaultdict(list)
    })
    
    for q_id, pred_data in predictions.items():
        q = next(q for q in dataset['questions'] if q['id'] == q_id)
        base_id = q['base_id']
        
        consistency_results[base_id][q['polarity']].append(pred_data['prediction'])
        consistency_results[base_id]['variations'][q['variation_num']].append(pred_data['prediction'])
    
    # Calculate consistency metrics
    metrics = {}
    for base_id, data in consistency_results.items():
        # Check if positive variations are consistent
        pos_preds = data['positive']
        neg_preds = data['negative']
        
        if pos_preds:
            pos_consistency = np.std([p > 0.5 for p in pos_preds]) == 0
            pos_avg = np.mean(pos_preds)
        else:
            pos_consistency = True
            pos_avg = 0.5
        
        if neg_preds:
            neg_consistency = np.std([p > 0.5 for p in neg_preds]) == 0
            neg_avg = np.mean(neg_preds)
        else:
            neg_consistency = True
            neg_avg = 0.5
        
        # Check if negations properly flip
        polarity_flip = abs(pos_avg - (1 - neg_avg)) < 0.2
        
        metrics[base_id] = {
            'positive_consistent': pos_consistency,
            'negative_consistent': neg_consistency,
            'polarity_flip_correct': polarity_flip,
            'positive_avg': pos_avg,
            'negative_avg': neg_avg
        }
    
    return metrics


def evaluate_model(model: MCDropoutModel, dataloader: DataLoader, 
                   questions: List[Dict], device: torch.device) -> Dict:
    """Evaluate model with uncertainty metrics."""
    predictions = {}
    
    model.eval()
    all_preds = []
    all_labels = []
    all_uncertainties = []
    
    with torch.no_grad():
        for i, (features, labels) in enumerate(dataloader):
            features = features.to(device)
            
            # Get predictions with uncertainty
            mean_pred, std_pred = model.predict_with_uncertainty(features)
            
            # Store results
            batch_size = features.size(0)
            start_idx = i * dataloader.batch_size
            
            for j in range(batch_size):
                q_idx = start_idx + j
                if q_idx < len(questions):
                    q = questions[q_idx]
                    predictions[q['id']] = {
                        'prediction': float(mean_pred[j]),
                        'uncertainty': float(std_pred[j]),
                        'true_label': float(labels[j])
                    }
            
            all_preds.extend(mean_pred.flatten())
            all_labels.extend(labels.numpy().flatten())
            all_uncertainties.extend(std_pred.flatten())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_uncertainties = np.array(all_uncertainties)
    
    # Binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    accuracy = (binary_preds == all_labels).mean()
    
    # Separate QC and viral questions
    qc_mask = np.array([q.get('is_qc', False) for q in questions])
    
    qc_acc = (binary_preds[qc_mask] == all_labels[qc_mask]).mean() if qc_mask.any() else 0
    viral_acc = (binary_preds[~qc_mask] == all_labels[~qc_mask]).mean() if (~qc_mask).any() else 0
    
    qc_uncertainty = all_uncertainties[qc_mask].mean() if qc_mask.any() else 0
    viral_uncertainty = all_uncertainties[~qc_mask].mean() if (~qc_mask).any() else 0
    
    metrics = {
        'overall_accuracy': accuracy,
        'qc_accuracy': qc_acc,
        'viral_accuracy': viral_acc,
        'qc_uncertainty': qc_uncertainty,
        'viral_uncertainty': viral_uncertainty,
        'predictions': predictions
    }
    
    return metrics


def train_model(model: MCDropoutModel, train_loader: DataLoader, val_loader: DataLoader,
                device: torch.device, epochs: int = 50) -> List[Dict]:
    """Train the MC Dropout model."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        scheduler.step(avg_val_loss)
        
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        })
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                       f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train MC Dropout on debates dataset")
    parser.add_argument("--dataset", default="data/debates_reasoning.json", 
                       help="Path to reasoning dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--output", default="models/mc_dropout_debates.pth", 
                       help="Output model path")
    
    args = parser.parse_args()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    
    # Filter questions with valid answers
    valid_questions = [q for q in dataset['questions'] 
                      if q.get('answer') in ['yes', 'no']]
    
    logger.info(f"Valid questions: {len(valid_questions)}/{len(dataset['questions'])}")
    
    # Split by base_id to ensure variations stay together
    base_ids = list(set(q['base_id'] for q in valid_questions))
    
    # Handle small datasets
    if len(base_ids) < 5:
        logger.warning(f"Small dataset with only {len(base_ids)} base questions. Using all for training.")
        train_questions = valid_questions
        val_questions = valid_questions[:len(valid_questions)//2]  # Use half for validation
        test_questions = valid_questions[len(valid_questions)//2:]  # Use other half for test
    else:
        train_base, test_base = train_test_split(base_ids, test_size=0.2, random_state=42)
        train_base, val_base = train_test_split(train_base, test_size=0.2, random_state=42)
        
        train_questions = [q for q in valid_questions if q['base_id'] in train_base]
        val_questions = [q for q in valid_questions if q['base_id'] in val_base]
        test_questions = [q for q in valid_questions if q['base_id'] in test_base]
    
    logger.info(f"Train: {len(train_questions)}, Val: {len(val_questions)}, Test: {len(test_questions)}")
    
    # Create datasets
    train_dataset = DebatesDataset(train_questions, fit_transform=True)
    val_dataset = DebatesDataset(val_questions, 
                                vectorizer=train_dataset.vectorizer,
                                scaler=train_dataset.scaler,
                                fit_transform=False)
    test_dataset = DebatesDataset(test_questions,
                                 vectorizer=train_dataset.vectorizer,
                                 scaler=train_dataset.scaler,
                                 fit_transform=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = train_dataset.features.shape[1]
    model = MCDropoutModel(input_dim, dropout_rate=args.dropout).to(device)
    
    logger.info(f"Model initialized with input dim: {input_dim}")
    
    # Train model
    logger.info("Training model...")
    history = train_model(model, train_loader, val_loader, device, epochs=args.epochs)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, test_questions, device)
    
    logger.info(f"Test Accuracy: {test_metrics['overall_accuracy']:.4f}")
    logger.info(f"QC Accuracy: {test_metrics['qc_accuracy']:.4f}")
    logger.info(f"Viral Accuracy: {test_metrics['viral_accuracy']:.4f}")
    logger.info(f"QC Uncertainty: {test_metrics['qc_uncertainty']:.4f}")
    logger.info(f"Viral Uncertainty: {test_metrics['viral_uncertainty']:.4f}")
    
    # Analyze consistency
    logger.info("Analyzing consistency across variations...")
    consistency_metrics = analyze_consistency(dataset, test_metrics['predictions'])
    
    consistent_count = sum(1 for m in consistency_metrics.values() 
                          if m['positive_consistent'] and m['negative_consistent'])
    flip_correct = sum(1 for m in consistency_metrics.values() 
                      if m['polarity_flip_correct'])
    
    logger.info(f"Consistent base questions: {consistent_count}/{len(consistency_metrics)}")
    logger.info(f"Correct polarity flips: {flip_correct}/{len(consistency_metrics)}")
    
    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vectorizer': train_dataset.vectorizer,
        'scaler': train_dataset.scaler,
        'input_dim': input_dim,
        'dropout_rate': args.dropout,
        'metrics': test_metrics,
        'consistency': consistency_metrics
    }, args.output)
    
    logger.info(f"Model saved to {args.output}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot([h['epoch'] for h in history], [h['train_loss'] for h in history], label='Train')
    plt.plot([h['epoch'] for h in history], [h['val_loss'] for h in history], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    plt.subplot(1, 2, 2)
    plt.plot([h['epoch'] for h in history], [h['val_acc'] for h in history])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('models/mc_dropout_training.png')
    logger.info("Training plots saved to models/mc_dropout_training.png")


if __name__ == "__main__":
    main()