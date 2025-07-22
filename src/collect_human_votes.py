#!/usr/bin/env python3
"""
Human Vote Collection Interface

Collects real human votes on ambiguous debate questions.
Can be run as a CLI tool or integrated into a web interface.
"""

import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import sys
from pathlib import Path

# For web interface
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class HumanVoteCollector:
    """Collect and manage human votes on debate questions."""
    
    def __init__(self, dataset_path: str, votes_path: str = None):
        self.dataset_path = dataset_path
        self.votes_path = votes_path or dataset_path.replace('.json', '_votes.json')
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Load or initialize votes
        self.votes = self._load_votes()
        
    def _load_dataset(self) -> Dict:
        """Load the debates dataset."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def _load_votes(self) -> Dict:
        """Load existing votes or initialize new structure."""
        if os.path.exists(self.votes_path):
            with open(self.votes_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "total_votes": 0,
                    "unique_voters": 0
                },
                "votes": {},  # question_id -> vote data
                "voter_sessions": []  # Track anonymous sessions
            }
    
    def save_votes(self):
        """Save votes to file."""
        self.votes["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.votes_path, 'w') as f:
            json.dump(self.votes, f, indent=2)
    
    def get_question_for_voting(self, prefer_unvoted: bool = True) -> Optional[Dict]:
        """Get a question for voting, preferring those with fewer votes."""
        samples = self.dataset["samples"]
        
        if prefer_unvoted:
            # Sort by number of votes (ascending)
            sorted_samples = sorted(
                samples,
                key=lambda s: self.votes["votes"].get(s["id"], {}).get("total", 0)
            )
            # Return from the least voted 20%
            candidate_pool = sorted_samples[:max(1, len(sorted_samples) // 5)]
        else:
            candidate_pool = samples
        
        return random.choice(candidate_pool) if candidate_pool else None
    
    def record_vote(self, question_id: str, vote: str, voter_id: str = None, 
                   confidence: float = None, reasoning: str = None) -> Dict:
        """Record a human vote."""
        if vote.lower() not in ['yes', 'no']:
            raise ValueError("Vote must be 'yes' or 'no'")
        
        # Initialize vote data for question if needed
        if question_id not in self.votes["votes"]:
            self.votes["votes"][question_id] = {
                "yes": 0,
                "no": 0,
                "total": 0,
                "votes_log": [],
                "confidence_scores": []
            }
        
        # Record vote
        vote_data = self.votes["votes"][question_id]
        vote_data[vote.lower()] += 1
        vote_data["total"] += 1
        
        # Log individual vote
        vote_entry = {
            "vote": vote.lower(),
            "timestamp": datetime.now().isoformat(),
            "voter_id": voter_id or f"anon_{datetime.now().timestamp()}",
            "confidence": confidence,
            "reasoning": reasoning
        }
        vote_data["votes_log"].append(vote_entry)
        
        if confidence is not None:
            vote_data["confidence_scores"].append(confidence)
        
        # Update metadata
        self.votes["metadata"]["total_votes"] += 1
        
        # Track voter session
        if voter_id and voter_id not in self.votes["voter_sessions"]:
            self.votes["voter_sessions"].append(voter_id)
            self.votes["metadata"]["unique_voters"] = len(self.votes["voter_sessions"])
        
        # Save after each vote
        self.save_votes()
        
        return vote_data
    
    def get_vote_statistics(self, question_id: str) -> Dict:
        """Get voting statistics for a question."""
        if question_id not in self.votes["votes"]:
            return {"yes": 0, "no": 0, "total": 0, "yes_percentage": 0.5}
        
        vote_data = self.votes["votes"][question_id]
        total = vote_data["total"]
        
        stats = {
            "yes": vote_data["yes"],
            "no": vote_data["no"],
            "total": total,
            "yes_percentage": vote_data["yes"] / total if total > 0 else 0.5,
            "no_percentage": vote_data["no"] / total if total > 0 else 0.5,
            "consensus_strength": abs(vote_data["yes"] - vote_data["no"]) / total if total > 0 else 0
        }
        
        if vote_data["confidence_scores"]:
            stats["avg_confidence"] = sum(vote_data["confidence_scores"]) / len(vote_data["confidence_scores"])
        
        return stats
    
    def update_dataset_with_votes(self):
        """Update the main dataset file with collected votes."""
        for sample in self.dataset["samples"]:
            question_id = sample["id"]
            if question_id in self.votes["votes"]:
                stats = self.get_vote_statistics(question_id)
                sample["human_votes"] = {
                    "status": "collected",
                    "yes": stats["yes"],
                    "no": stats["no"],
                    "total": stats["total"],
                    "yes_percentage": stats["yes_percentage"],
                    "consensus_strength": stats["consensus_strength"]
                }
        
        # Save updated dataset
        with open(self.dataset_path, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        
        print(f"Updated dataset with votes: {self.dataset_path}")


def cli_voting_interface(collector: HumanVoteCollector):
    """Command-line interface for collecting votes."""
    print("\n" + "="*60)
    print("HUMAN VOTE COLLECTION - Ambiguous Debates")
    print("="*60)
    print("\nFor each question, answer YES or NO.")
    print("Type 'quit' to exit, 'skip' to skip a question.\n")
    
    voter_id = f"cli_user_{datetime.now().timestamp()}"
    votes_collected = 0
    
    while True:
        # Get a question
        question_data = collector.get_question_for_voting()
        if not question_data:
            print("No more questions available!")
            break
        
        # Display question
        print("\n" + "-"*60)
        print(f"Category: {question_data['category'].replace('_', ' ').title()}")
        print(f"\nQuestion: {question_data['prompt']}")
        
        # Show current stats
        stats = collector.get_vote_statistics(question_data['id'])
        if stats['total'] > 0:
            print(f"\nCurrent votes: Yes: {stats['yes_percentage']:.1%} | No: {stats['no_percentage']:.1%} (n={stats['total']})")
        
        # Get vote
        while True:
            vote = input("\nYour answer (yes/no/skip/quit): ").strip().lower()
            
            if vote == 'quit':
                print(f"\nThanks for voting! You cast {votes_collected} votes.")
                return
            
            if vote == 'skip':
                break
            
            if vote in ['yes', 'no', 'y', 'n']:
                vote = 'yes' if vote.startswith('y') else 'no'
                
                # Optional: get confidence
                confidence = None
                conf_input = input("How confident are you? (1-5, or press Enter to skip): ").strip()
                if conf_input and conf_input.isdigit():
                    confidence = int(conf_input) / 5.0  # Normalize to 0-1
                
                # Optional: get reasoning
                reasoning = input("Brief reasoning (optional, press Enter to skip): ").strip()
                
                # Record vote
                collector.record_vote(
                    question_data['id'],
                    vote,
                    voter_id,
                    confidence,
                    reasoning if reasoning else None
                )
                
                votes_collected += 1
                print(f"✓ Vote recorded! ({votes_collected} total)")
                break
            else:
                print("Please enter 'yes', 'no', 'skip', or 'quit'.")


def streamlit_interface(collector: HumanVoteCollector):
    """Streamlit web interface for collecting votes."""
    st.set_page_config(page_title="Debate Voting", page_icon="🤔")
    
    st.title("🤔 Ambiguous Debates - Human Voting")
    st.markdown("Help us understand how people think about these debatable questions!")
    
    # Session state
    if 'voter_id' not in st.session_state:
        st.session_state.voter_id = f"web_user_{datetime.now().timestamp()}"
    if 'votes_cast' not in st.session_state:
        st.session_state.votes_cast = 0
    
    # Get question
    question = collector.get_question_for_voting()
    if not question:
        st.error("No questions available!")
        return
    
    # Display question
    st.markdown("---")
    category_display = question['category'].replace('_', ' ').title()
    st.markdown(f"**Category:** {category_display}")
    
    st.markdown("### " + question['prompt'])
    
    # Show current voting stats
    stats = collector.get_vote_statistics(question['id'])
    if stats['total'] > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Yes", f"{stats['yes_percentage']:.1%}")
        with col2:
            st.metric("No", f"{stats['no_percentage']:.1%}")
        with col3:
            st.metric("Total Votes", stats['total'])
    
    # Voting interface
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ YES", use_container_width=True, type="primary"):
            collector.record_vote(question['id'], 'yes', st.session_state.voter_id)
            st.session_state.votes_cast += 1
            st.rerun()
    
    with col2:
        if st.button("❌ NO", use_container_width=True, type="primary"):
            collector.record_vote(question['id'], 'no', st.session_state.voter_id)
            st.session_state.votes_cast += 1
            st.rerun()
    
    if st.button("Skip This Question →"):
        st.rerun()
    
    # Show reasoning if available
    if question.get('reasoning_trace'):
        with st.expander("🤖 AI Reasoning"):
            for token in question['reasoning_trace']['tokens'][:5]:  # Show first 5 steps
                st.text(f"• {token}")
            if len(question['reasoning_trace']['tokens']) > 5:
                st.text(f"... and {len(question['reasoning_trace']['tokens']) - 5} more steps")
    
    # Stats
    st.sidebar.markdown(f"### Your Stats")
    st.sidebar.markdown(f"Votes cast: **{st.session_state.votes_cast}**")
    
    # Overall stats
    st.sidebar.markdown("### Overall Stats")
    st.sidebar.markdown(f"Total votes: **{collector.votes['metadata']['total_votes']}**")
    st.sidebar.markdown(f"Unique voters: **{collector.votes['metadata']['unique_voters']}**")


def main():
    parser = argparse.ArgumentParser(description="Collect human votes on debate questions")
    parser.add_argument("--dataset", type=str, default="data/debates_dataset.json",
                       help="Path to debates dataset")
    parser.add_argument("--interface", choices=["cli", "web"], default="cli",
                       help="Interface type (cli or web)")
    parser.add_argument("--update-dataset", action="store_true",
                       help="Update main dataset with collected votes")
    
    args = parser.parse_args()
    
    # Initialize collector
    try:
        collector = HumanVoteCollector(args.dataset)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {args.dataset}")
        print("Please run generate_debates_dataset.py first")
        sys.exit(1)
    
    # Update dataset if requested
    if args.update_dataset:
        collector.update_dataset_with_votes()
        return
    
    # Run interface
    if args.interface == "web":
        if not STREAMLIT_AVAILABLE:
            print("Streamlit not installed. Install with: pip install streamlit")
            sys.exit(1)
        streamlit_interface(collector)
    else:
        cli_voting_interface(collector)


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE and len(sys.argv) == 1:
        # If run without args and streamlit available, default to web
        streamlit_interface(HumanVoteCollector("data/debates_dataset.json"))
    else:
        main()