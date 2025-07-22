#!/usr/bin/env python3
"""
Evidence Extraction from DeepSeek Reasoning

Extracts and analyzes evidence accumulation from DeepSeek reasoning traces
for visualization and model training.
"""

import re
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EvidenceToken:
    """Represents a reasoning token with its evidence contribution."""
    text: str
    position: int
    evidence_score: float
    evidence_delta: float  # Change from previous
    confidence_level: str  # high, medium, low
    evidence_type: str  # supporting, opposing, neutral, uncertainty


class EvidenceExtractor:
    """Extract evidence patterns from DeepSeek reasoning traces."""
    
    # Evidence indicator patterns
    STRONG_POSITIVE = [
        r'\b(clearly|definitely|obviously|certainly|absolutely|undoubtedly)\b',
        r'\b(proves?|confirms?|demonstrates?|establishes?)\b',
        r'\b(must be|has to be|can only be)\b'
    ]
    
    MODERATE_POSITIVE = [
        r'\b(suggests?|indicates?|shows?|supports?|implies?)\b',
        r'\b(likely|probably|appears? to|seems? to)\b',
        r'\b(evidence points? to|leans? toward)\b'
    ]
    
    WEAK_POSITIVE = [
        r'\b(might|could|possibly|perhaps|maybe)\b',
        r'\b(somewhat|slightly|marginally)\b',
        r'\b(tends? to|often|usually)\b'
    ]
    
    UNCERTAINTY = [
        r'\b(however|but|although|though|yet|nevertheless)\b',
        r'\b(unclear|uncertain|ambiguous|debatable)\b',
        r'\b(depends? on|varies?|subjective)\b',
        r'\b(on the other hand|alternatively|conversely)\b'
    ]
    
    NEGATIVE = [
        r'\b(not|no|neither|nor|never)\b',
        r'\b(contradicts?|opposes?|against|refuses?)\b',
        r'\b(incorrect|wrong|false|mistaken)\b',
        r'\b(doesn\'t|isn\'t|aren\'t|won\'t|can\'t)\b'
    ]
    
    def __init__(self):
        # Compile patterns for efficiency
        self.patterns = {
            'strong_positive': [re.compile(p, re.IGNORECASE) for p in self.STRONG_POSITIVE],
            'moderate_positive': [re.compile(p, re.IGNORECASE) for p in self.MODERATE_POSITIVE],
            'weak_positive': [re.compile(p, re.IGNORECASE) for p in self.WEAK_POSITIVE],
            'uncertainty': [re.compile(p, re.IGNORECASE) for p in self.UNCERTAINTY],
            'negative': [re.compile(p, re.IGNORECASE) for p in self.NEGATIVE]
        }
    
    def extract_evidence(self, reasoning_trace: Dict, final_answer: str) -> Dict:
        """Extract evidence from a complete reasoning trace."""
        tokens = reasoning_trace.get('tokens', [])
        
        # Determine answer polarity
        answer_polarity = 1.0 if final_answer.lower() == 'yes' else -1.0
        
        # Extract evidence for each token
        evidence_tokens = []
        cumulative_score = 0.0
        
        for i, token_text in enumerate(tokens):
            # Analyze token
            token_evidence = self._analyze_token(token_text, answer_polarity)
            
            # Calculate cumulative score
            cumulative_score += token_evidence['delta']
            cumulative_score = max(-1.0, min(1.0, cumulative_score))  # Clamp to [-1, 1]
            
            # Create evidence token
            evidence_token = EvidenceToken(
                text=token_text,
                position=i,
                evidence_score=cumulative_score,
                evidence_delta=token_evidence['delta'],
                confidence_level=token_evidence['confidence'],
                evidence_type=token_evidence['type']
            )
            
            evidence_tokens.append(evidence_token)
        
        # Extract key features
        features = self._extract_reasoning_features(evidence_tokens)
        
        return {
            'tokens': evidence_tokens,
            'final_score': cumulative_score,
            'features': features,
            'evidence_flow': self._analyze_evidence_flow(evidence_tokens)
        }
    
    def _analyze_token(self, token_text: str, answer_polarity: float) -> Dict:
        """Analyze a single token for evidence indicators."""
        # Check for pattern matches
        matches = defaultdict(int)
        
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(token_text):
                    matches[pattern_type] += 1
        
        # Determine evidence strength and type
        if matches['uncertainty'] > 0:
            delta = -0.1 * answer_polarity  # Uncertainty reduces confidence
            confidence = 'low'
            evidence_type = 'uncertainty'
        elif matches['negative'] > 0:
            # Check if negation supports or opposes
            if self._is_double_negative(token_text):
                delta = 0.2 * answer_polarity
                confidence = 'medium'
                evidence_type = 'supporting'
            else:
                delta = -0.2 * answer_polarity
                confidence = 'medium'
                evidence_type = 'opposing'
        elif matches['strong_positive'] > 0:
            delta = 0.3 * answer_polarity
            confidence = 'high'
            evidence_type = 'supporting'
        elif matches['moderate_positive'] > 0:
            delta = 0.2 * answer_polarity
            confidence = 'medium'
            evidence_type = 'supporting'
        elif matches['weak_positive'] > 0:
            delta = 0.1 * answer_polarity
            confidence = 'low'
            evidence_type = 'supporting'
        else:
            # Neutral token
            delta = 0.05 * answer_polarity
            confidence = 'low'
            evidence_type = 'neutral'
        
        return {
            'delta': delta,
            'confidence': confidence,
            'type': evidence_type,
            'matches': dict(matches)
        }
    
    def _is_double_negative(self, text: str) -> bool:
        """Check if text contains double negative (which becomes positive)."""
        negative_count = len(re.findall(r'\b(not|no|never|neither)\b', text, re.IGNORECASE))
        return negative_count >= 2
    
    def _extract_reasoning_features(self, evidence_tokens: List[EvidenceToken]) -> Dict:
        """Extract high-level features from evidence tokens."""
        if not evidence_tokens:
            return {}
        
        # Count evidence types
        type_counts = defaultdict(int)
        confidence_counts = defaultdict(int)
        
        for token in evidence_tokens:
            type_counts[token.evidence_type] += 1
            confidence_counts[token.confidence_level] += 1
        
        # Calculate statistics
        scores = [t.evidence_score for t in evidence_tokens]
        deltas = [t.evidence_delta for t in evidence_tokens]
        
        features = {
            'num_tokens': len(evidence_tokens),
            'final_score': evidence_tokens[-1].evidence_score,
            'max_score': max(scores),
            'min_score': min(scores),
            'score_variance': np.var(scores),
            'avg_delta': np.mean(deltas),
            'direction_changes': self._count_direction_changes(deltas),
            'evidence_types': dict(type_counts),
            'confidence_distribution': dict(confidence_counts),
            'uncertainty_ratio': type_counts['uncertainty'] / len(evidence_tokens),
            'high_confidence_ratio': confidence_counts['high'] / len(evidence_tokens)
        }
        
        return features
    
    def _count_direction_changes(self, deltas: List[float]) -> int:
        """Count how many times evidence direction changes."""
        if len(deltas) < 2:
            return 0
        
        changes = 0
        for i in range(1, len(deltas)):
            if np.sign(deltas[i]) != np.sign(deltas[i-1]) and deltas[i] != 0 and deltas[i-1] != 0:
                changes += 1
        
        return changes
    
    def _analyze_evidence_flow(self, evidence_tokens: List[EvidenceToken]) -> Dict:
        """Analyze the flow and progression of evidence."""
        if not evidence_tokens:
            return {}
        
        scores = [t.evidence_score for t in evidence_tokens]
        
        # Find key turning points
        turning_points = []
        for i in range(1, len(scores) - 1):
            # Local maxima or minima
            if (scores[i] > scores[i-1] and scores[i] > scores[i+1]) or \
               (scores[i] < scores[i-1] and scores[i] < scores[i+1]):
                turning_points.append({
                    'position': i,
                    'score': scores[i],
                    'token': evidence_tokens[i].text[:50] + '...' if len(evidence_tokens[i].text) > 50 else evidence_tokens[i].text
                })
        
        # Identify phases (building, stable, declining)
        phases = self._identify_phases(scores)
        
        return {
            'turning_points': turning_points,
            'phases': phases,
            'monotonic': len(turning_points) == 0,
            'final_direction': 'positive' if scores[-1] > 0 else 'negative',
            'peak_position': np.argmax(np.abs(scores)),
            'convergence_rate': self._calculate_convergence_rate(scores)
        }
    
    def _identify_phases(self, scores: List[float]) -> List[Dict]:
        """Identify distinct phases in evidence accumulation."""
        if len(scores) < 3:
            return []
        
        phases = []
        phase_start = 0
        current_trend = None
        
        for i in range(1, len(scores)):
            # Calculate local trend
            if scores[i] > scores[i-1] + 0.05:
                trend = 'increasing'
            elif scores[i] < scores[i-1] - 0.05:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Check for phase change
            if current_trend is None:
                current_trend = trend
            elif trend != current_trend:
                # Phase ended
                phases.append({
                    'start': phase_start,
                    'end': i - 1,
                    'trend': current_trend,
                    'score_change': scores[i-1] - scores[phase_start]
                })
                phase_start = i
                current_trend = trend
        
        # Add final phase
        if current_trend:
            phases.append({
                'start': phase_start,
                'end': len(scores) - 1,
                'trend': current_trend,
                'score_change': scores[-1] - scores[phase_start]
            })
        
        return phases
    
    def _calculate_convergence_rate(self, scores: List[float]) -> float:
        """Calculate how quickly evidence converges to final conclusion."""
        if len(scores) < 2:
            return 0.0
        
        final_score = scores[-1]
        if final_score == 0:
            return 0.0
        
        # Find when we reach 80% of final score
        threshold = final_score * 0.8
        
        for i, score in enumerate(scores):
            if abs(score) >= abs(threshold):
                return i / len(scores)  # Fraction of reasoning needed
        
        return 1.0  # Never converged
    
    def format_for_visualization(self, evidence_data: Dict) -> Dict:
        """Format evidence data for visualization in the app."""
        tokens = evidence_data['tokens']
        
        return {
            'token_texts': [t.text for t in tokens],
            'evidence_scores': [t.evidence_score for t in tokens],
            'evidence_deltas': [t.evidence_delta for t in tokens],
            'confidence_levels': [t.confidence_level for t in tokens],
            'evidence_types': [t.evidence_type for t in tokens],
            'features': evidence_data['features'],
            'flow_analysis': evidence_data['evidence_flow']
        }


def extract_evidence_from_dataset(dataset_path: str, output_path: str = None):
    """Extract evidence from all samples in a dataset."""
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    extractor = EvidenceExtractor()
    
    # Process each sample
    for sample in dataset['samples']:
        if 'reasoning_trace' in sample:
            evidence_data = extractor.extract_evidence(
                sample['reasoning_trace'],
                sample['model_answer']
            )
            
            # Add formatted version for visualization
            sample['evidence_analysis'] = extractor.format_for_visualization(evidence_data)
            
            # Update evidence scores in reasoning trace
            sample['reasoning_trace']['evidence_scores'] = [
                t.evidence_score for t in evidence_data['tokens']
            ]
    
    # Save updated dataset
    output_path = output_path or dataset_path.replace('.json', '_with_evidence.json')
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Extracted evidence for {len(dataset['samples'])} samples")
    print(f"Saved to: {output_path}")
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract evidence from reasoning traces")
    parser.add_argument("--dataset", type=str, default="data/debates_dataset.json",
                       help="Path to dataset")
    parser.add_argument("--output", type=str, help="Output path (optional)")
    
    args = parser.parse_args()
    
    extract_evidence_from_dataset(args.dataset, args.output)