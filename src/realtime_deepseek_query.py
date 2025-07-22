#!/usr/bin/env python3
"""
Real-time DeepSeek Query Support

Enables real-time querying of DeepSeek for custom questions
with evidence extraction and visualization support.
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extract_reasoning_evidence import EvidenceExtractor


class RealtimeDeepSeekQuery:
    """Handle real-time queries to DeepSeek API with caching."""
    
    def __init__(self, api_key: str = None, cache_dir: str = "cache/deepseek_queries"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.evidence_extractor = EvidenceExtractor()
    
    def _get_cache_key(self, question: str, category: str = "custom") -> str:
        """Generate cache key for a question."""
        content = f"{question}_{category}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached response if available."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_path):
            # Check if cache is fresh (less than 24 hours old)
            file_age = time.time() - os.path.getmtime(cache_path)
            if file_age < 24 * 3600:  # 24 hours
                with open(cache_path, 'r') as f:
                    return json.load(f)
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save response to cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def query_debate_question(
        self,
        question: str,
        category: str = "custom",
        use_cache: bool = True,
        stream_callback = None
    ) -> Dict:
        """
        Query DeepSeek for a debate question with reasoning.
        
        Args:
            question: The binary question to analyze
            category: Question category for context
            use_cache: Whether to use cached responses
            stream_callback: Optional callback for streaming responses
        
        Returns:
            Dict with reasoning trace, answer, and evidence analysis
        """
        # Check cache first
        cache_key = self._get_cache_key(question, category)
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached:
                return cached
        
        # Prepare system prompt
        system_prompt = f"""You are analyzing an ambiguous binary question where reasonable people might disagree.
Category: {category}

Provide step-by-step reasoning that:
1. Considers multiple perspectives
2. Identifies factors that create uncertainty
3. Acknowledges why people might disagree
4. Reaches a conclusion while noting the ambiguity

Format your response as clear reasoning steps, each on a new line.
End with your binary answer: "Yes" or "No"."""
        
        user_prompt = f"""Question: {question}

Provide detailed reasoning considering all perspectives, then give your final answer."""
        
        try:
            # Make API call
            if stream_callback:
                # Streaming mode
                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                    stream=True
                )
                
                # Collect streamed response
                full_content = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_content += content
                        if stream_callback:
                            stream_callback(content)
                
                content = full_content
            else:
                # Non-streaming mode
                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                content = response.choices[0].message.content
            
            # Parse response
            lines = content.strip().split('\n')
            reasoning_tokens = []
            final_answer = None
            uncertainty_factors = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is the final answer
                if line.lower() in ['yes', 'no']:
                    final_answer = line.lower().capitalize()
                elif any(word in line.lower() for word in ['however', 'but', 'although', 'on the other hand']):
                    uncertainty_factors.append(line)
                    reasoning_tokens.append(line)
                else:
                    reasoning_tokens.append(line)
            
            # If no clear answer found, default to No
            if not final_answer:
                final_answer = "No"
            
            # Create reasoning trace
            reasoning_trace = {
                "tokens": reasoning_tokens,
                "uncertainty_factors": uncertainty_factors
            }
            
            # Extract evidence
            evidence_data = self.evidence_extractor.extract_evidence(
                reasoning_trace,
                final_answer
            )
            
            # Format result
            result = {
                "question": question,
                "category": category,
                "model_answer": final_answer,
                "reasoning_trace": reasoning_trace,
                "evidence_analysis": self.evidence_extractor.format_for_visualization(evidence_data),
                "api_response": content,
                "timestamp": datetime.now().isoformat(),
                "cached": False
            }
            
            # Cache result
            if use_cache:
                self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            print(f"API error: {e}")
            return {
                "question": question,
                "category": category,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_query(self, questions: List[Dict], use_cache: bool = True) -> List[Dict]:
        """Query multiple questions with rate limiting."""
        results = []
        
        for i, q_data in enumerate(questions):
            question = q_data.get('question', q_data.get('prompt', ''))
            category = q_data.get('category', 'custom')
            
            print(f"Processing {i+1}/{len(questions)}: {question[:50]}...")
            
            result = self.query_debate_question(question, category, use_cache)
            results.append(result)
            
            # Rate limiting
            if i < len(questions) - 1:
                time.sleep(1)  # 1 second between requests
        
        return results
    
    def analyze_custom_question(self, question: str) -> Dict:
        """
        Analyze a custom question and return visualization-ready data.
        
        This is the main method for the visual app to use.
        """
        # Determine likely category
        category = self._infer_category(question)
        
        # Query DeepSeek
        result = self.query_debate_question(question, category)
        
        if 'error' in result:
            return result
        
        # Add visualization metadata
        result['visualization'] = {
            'ready': True,
            'evidence_chart_data': {
                'x': list(range(len(result['reasoning_trace']['tokens']))),
                'y': result['evidence_analysis']['evidence_scores'],
                'labels': [f"Step {i+1}" for i in range(len(result['reasoning_trace']['tokens']))],
                'hover_text': result['reasoning_trace']['tokens']
            },
            'uncertainty_level': self._calculate_uncertainty_level(result['evidence_analysis']),
            'debate_intensity': len(result['reasoning_trace']['uncertainty_factors']) / len(result['reasoning_trace']['tokens'])
        }
        
        return result
    
    def _infer_category(self, question: str) -> str:
        """Infer the likely category of a question."""
        question_lower = question.lower()
        
        # Simple keyword-based categorization
        if any(word in question_lower for word in ['food', 'eat', 'sandwich', 'soup', 'pizza']):
            return 'food_classification'
        elif any(word in question_lower for word in ['movie', 'christmas', 'sport', 'acceptable']):
            return 'social_cultural'
        elif any(word in question_lower for word in ['tall', 'warm', 'expensive', 'far', 'old']):
            return 'threshold_questions'
        elif any(word in question_lower for word in ['technically', 'classify', 'count as']):
            return 'edge_classifications'
        else:
            return 'custom'
    
    def _calculate_uncertainty_level(self, evidence_analysis: Dict) -> str:
        """Calculate overall uncertainty level from evidence analysis."""
        features = evidence_analysis.get('features', {})
        
        uncertainty_ratio = features.get('uncertainty_ratio', 0)
        direction_changes = features.get('direction_changes', 0)
        
        if uncertainty_ratio > 0.3 or direction_changes > 3:
            return 'high'
        elif uncertainty_ratio > 0.1 or direction_changes > 1:
            return 'medium'
        else:
            return 'low'


def demo_realtime_query():
    """Demo function showing real-time query usage."""
    # Initialize querier
    querier = RealtimeDeepSeekQuery()
    
    # Example questions
    test_questions = [
        "Is a hot dog a sandwich?",
        "Is water wet?",
        "Is 6 feet tall for a man?",
        "Is chess a sport?"
    ]
    
    print("Real-time DeepSeek Query Demo")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        
        # Query with streaming
        def stream_callback(content):
            print(content, end='', flush=True)
        
        result = querier.analyze_custom_question(question)
        
        if 'error' not in result:
            print(f"\n\nAnswer: {result['model_answer']}")
            print(f"Uncertainty Level: {result['visualization']['uncertainty_level']}")
            print(f"Debate Intensity: {result['visualization']['debate_intensity']:.2%}")
        else:
            print(f"\nError: {result['error']}")
        
        time.sleep(2)  # Pause between questions


if __name__ == "__main__":
    demo_realtime_query()