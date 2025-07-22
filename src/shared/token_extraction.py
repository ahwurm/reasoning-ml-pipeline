"""
Shared token extraction utilities for R1 reasoning.

This module provides common functionality used by both API and local extractors.
"""
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def extract_reasoning_tokens(text: str) -> List[str]:
    """
    Extract reasoning tokens from <think> tags.
    
    Args:
        text: Response text containing <think> tags
        
    Returns:
        List of reasoning tokens
    """
    # Find all content within <think> tags
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL)
    
    reasoning_tokens = []
    for match in matches:
        # Split by newlines and clean up
        tokens = [
            token.strip() 
            for token in match.strip().split('\n') 
            if token.strip()
        ]
        reasoning_tokens.extend(tokens)
    
    return reasoning_tokens


def extract_final_answer(text: str) -> str:
    """
    Extract final answer (everything after last </think> tag).
    
    Args:
        text: Full response text
        
    Returns:
        Final answer text
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def parse_reasoning_structure(text: str) -> Dict[str, any]:
    """
    Parse reasoning structure with more detailed analysis.
    
    Args:
        text: Response text with reasoning
        
    Returns:
        Dictionary with structured reasoning data
    """
    # Extract all think blocks
    think_blocks = []
    think_pattern = r'<think>(.*?)</think>'
    
    for match in re.finditer(think_pattern, text, re.DOTALL):
        block_content = match.group(1).strip()
        block_tokens = [
            token.strip() 
            for token in block_content.split('\n') 
            if token.strip()
        ]
        
        think_blocks.append({
            "start_pos": match.start(),
            "end_pos": match.end(),
            "content": block_content,
            "tokens": block_tokens,
            "token_count": len(block_tokens)
        })
    
    # Extract sections between think blocks
    sections = []
    last_end = 0
    
    for block in think_blocks:
        # Content before this block
        if block["start_pos"] > last_end:
            pre_content = text[last_end:block["start_pos"]].strip()
            if pre_content:
                sections.append({
                    "type": "pre_reasoning",
                    "content": pre_content,
                    "position": last_end
                })
        
        # The think block itself
        sections.append({
            "type": "reasoning",
            "content": block["content"],
            "tokens": block["tokens"],
            "position": block["start_pos"]
        })
        
        last_end = block["end_pos"]
    
    # Final content after last think block
    if last_end < len(text):
        final_content = text[last_end:].strip()
        if final_content:
            sections.append({
                "type": "final_answer",
                "content": final_content,
                "position": last_end
            })
    
    # Combine all reasoning tokens
    all_tokens = []
    for block in think_blocks:
        all_tokens.extend(block["tokens"])
    
    return {
        "think_blocks": think_blocks,
        "sections": sections,
        "reasoning_tokens": all_tokens,
        "final_answer": extract_final_answer(text),
        "num_think_blocks": len(think_blocks),
        "total_reasoning_tokens": len(all_tokens)
    }


def validate_reasoning_response(text: str) -> Tuple[bool, List[str]]:
    """
    Validate that response contains proper reasoning structure.
    
    Args:
        text: Response text to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for think tags
    if "<think>" not in text:
        issues.append("No <think> tags found")
    
    if "</think>" not in text:
        issues.append("No closing </think> tag found")
    
    # Check for mismatched tags
    open_count = text.count("<think>")
    close_count = text.count("</think>")
    
    if open_count != close_count:
        issues.append(f"Mismatched think tags: {open_count} open, {close_count} close")
    
    # Check for empty think blocks
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL)
    
    for i, match in enumerate(matches):
        if not match.strip():
            issues.append(f"Empty think block at position {i+1}")
    
    # Check for final answer
    if "</think>" in text:
        final = text.split("</think>")[-1].strip()
        if not final:
            issues.append("No final answer after reasoning")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def format_reasoning_output(
    prompt: str,
    reasoning_tokens: List[str],
    final_answer: str,
    metadata: Optional[Dict] = None
) -> Dict[str, any]:
    """
    Format extraction results in consistent structure.
    
    Args:
        prompt: Original prompt
        reasoning_tokens: Extracted reasoning tokens
        final_answer: Final answer text
        metadata: Optional metadata
        
    Returns:
        Formatted result dictionary
    """
    result = {
        "prompt": prompt,
        "reasoning_tokens": reasoning_tokens,
        "final_answer": final_answer,
        "num_reasoning_tokens": len(reasoning_tokens),
        "timestamp": datetime.now().isoformat()
    }
    
    if metadata:
        result["metadata"] = metadata
    
    return result


def clean_reasoning_token(token: str) -> str:
    """
    Clean and normalize a reasoning token.
    
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
    token = re.sub(r'#+\s*', '', token)  # Remove headers
    
    # Normalize quotes
    token = token.replace('"', '"').replace('"', '"')
    token = token.replace(''', "'").replace(''', "'")
    
    # Remove leading/trailing punctuation that's not meaningful
    token = token.strip('.,;:')
    
    return token.strip()


def classify_token_type(token: str) -> str:
    """
    Classify the type of reasoning token.
    
    Args:
        token: Reasoning token text
        
    Returns:
        Token type classification
    """
    token_lower = token.lower()
    
    # Check for calculation/math
    if re.search(r'\d+\s*[+\-*/=]\s*\d+', token) or re.search(r'=\s*\d+', token):
        return "calculation"
    
    # Check for question
    if token.strip().endswith('?'):
        return "question"
    
    # Check for conclusion
    if any(word in token_lower for word in [
        'therefore', 'thus', 'so', 'in conclusion', 
        'finally', 'the answer is', 'this means'
    ]):
        return "conclusion"
    
    # Check for uncertainty
    if any(word in token_lower for word in [
        'maybe', 'perhaps', 'might', 'could', 
        'possibly', 'unsure', 'not certain'
    ]):
        return "uncertainty"
    
    # Check for contradiction/revision
    if any(word in token_lower for word in [
        'but', 'however', 'actually', 'wait',
        'no,', 'incorrect', 'wrong'
    ]):
        return "revision"
    
    # Check for setup/context
    if any(word in token_lower for word in [
        'let me', 'first', 'we need to', 'given',
        'we have', 'the problem'
    ]):
        return "setup"
    
    # Default
    return "statement"


def analyze_reasoning_pattern(tokens: List[str]) -> Dict[str, any]:
    """
    Analyze the pattern of reasoning tokens.
    
    Args:
        tokens: List of reasoning tokens
        
    Returns:
        Pattern analysis
    """
    if not tokens:
        return {
            "total_tokens": 0,
            "pattern": "empty",
            "has_conclusion": False,
            "has_revision": False,
            "dominant_type": None
        }
    
    # Classify all tokens
    token_types = [classify_token_type(token) for token in tokens]
    
    # Count types
    type_counts = {}
    for t in token_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    
    # Find dominant type
    dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
    
    # Check for specific patterns
    has_conclusion = "conclusion" in token_types
    has_revision = "revision" in token_types
    has_calculation = "calculation" in token_types
    has_uncertainty = "uncertainty" in token_types
    
    # Determine overall pattern
    if has_calculation and has_conclusion:
        pattern = "mathematical_proof"
    elif has_revision:
        pattern = "self_correction"
    elif has_uncertainty:
        pattern = "exploratory"
    elif has_conclusion and not has_calculation:
        pattern = "logical_argument"
    else:
        pattern = "sequential"
    
    return {
        "total_tokens": len(tokens),
        "type_distribution": type_counts,
        "dominant_type": dominant_type,
        "pattern": pattern,
        "has_conclusion": has_conclusion,
        "has_revision": has_revision,
        "has_calculation": has_calculation,
        "has_uncertainty": has_uncertainty,
        "conclusion_position": tokens.index(next((t for t, typ in zip(tokens, token_types) if typ == "conclusion"), None)) if has_conclusion else None
    }