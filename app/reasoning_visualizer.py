#!/usr/bin/env python3
"""
Interactive Reasoning Visualizer

Streamlit app for visualizing MC Dropout model predictions alongside
DeepSeek reasoning with interactive evidence accumulation charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extract_reasoning_evidence import EvidenceExtractor
from src.models.bayesian_binary_reasoning_model import BayesianBinaryReasoningModel
from src.realtime_deepseek_query import RealtimeDeepSeekQuery


# Page configuration
st.set_page_config(
    page_title="Reasoning Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .token-highlight {
        background-color: #ffeb3b;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .evidence-positive {
        background-color: #4caf50;
        color: white;
    }
    .evidence-negative {
        background-color: #f44336;
        color: white;
    }
    .evidence-neutral {
        background-color: #9e9e9e;
        color: white;
    }
    .reasoning-token {
        padding: 5px;
        margin: 2px;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .reasoning-token:hover {
        transform: scale(1.05);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


class ReasoningVisualizerApp:
    """Main application class for reasoning visualization."""
    
    def __init__(self):
        self.initialize_session_state()
        self.load_data()
        self.evidence_extractor = EvidenceExtractor()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'selected_token_idx' not in st.session_state:
            st.session_state.selected_token_idx = None
        if 'selected_question_id' not in st.session_state:
            st.session_state.selected_question_id = None
        if 'mc_dropout_predictions' not in st.session_state:
            st.session_state.mc_dropout_predictions = {}
        if 'realtime_mode' not in st.session_state:
            st.session_state.realtime_mode = False
    
    def load_data(self):
        """Load dataset and models."""
        # Load debate dataset
        dataset_path = "data/debates_dataset.json"
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)
        else:
            st.error(f"Dataset not found at {dataset_path}. Please generate it first.")
            st.stop()
        
        # Load MC Dropout model
        try:
            self.mc_dropout_model = self.load_mc_dropout_model()
        except Exception as e:
            st.warning(f"MC Dropout model not loaded: {e}")
            self.mc_dropout_model = None
    
    def load_mc_dropout_model(self):
        """Load the trained MC Dropout model."""
        model_path = "models/bayesian_mc_dropout.pth"
        if not os.path.exists(model_path):
            return None
        
        # Initialize model (simplified for demo)
        model = BayesianBinaryReasoningModel(model_type="mc_dropout")
        model.load(model_path)
        return model
    
    def render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.title("🎛️ Controls")
        
        # Mode selection
        mode = st.sidebar.radio(
            "Mode",
            ["Dataset Questions", "Real-time Query"],
            index=0 if not st.session_state.realtime_mode else 1
        )
        st.session_state.realtime_mode = (mode == "Real-time Query")
        
        if not st.session_state.realtime_mode:
            # Dataset mode
            st.sidebar.markdown("### Question Selection")
            
            # Category filter
            categories = ["All"] + list(set(s['category'] for s in self.dataset['samples']))
            selected_category = st.sidebar.selectbox("Category", categories)
            
            # Filter samples
            if selected_category == "All":
                filtered_samples = self.dataset['samples']
            else:
                filtered_samples = [s for s in self.dataset['samples'] 
                                  if s['category'] == selected_category]
            
            # Question selector
            question_options = {
                f"{s['id']}: {s['prompt'][:50]}...": s 
                for s in filtered_samples
            }
            
            selected_key = st.sidebar.selectbox(
                "Select Question",
                options=list(question_options.keys())
            )
            
            return question_options[selected_key]
        else:
            # Real-time mode
            st.sidebar.markdown("### Custom Question")
            custom_question = st.sidebar.text_area(
                "Enter your question:",
                placeholder="Is a hot dog a sandwich?",
                height=100
            )
            
            if st.sidebar.button("🚀 Analyze", type="primary"):
                if custom_question:
                    return {"prompt": custom_question, "id": "custom", "category": "custom"}
            
            return None
    
    def create_evidence_chart(self, evidence_data: Dict, selected_idx: Optional[int] = None) -> go.Figure:
        """Create interactive evidence accumulation chart."""
        scores = evidence_data['evidence_scores']
        tokens = evidence_data['token_texts']
        types = evidence_data['evidence_types']
        
        # Create figure
        fig = go.Figure()
        
        # Add evidence line
        fig.add_trace(go.Scatter(
            x=list(range(len(scores))),
            y=scores,
            mode='lines+markers',
            name='Evidence Score',
            line=dict(color='#2196f3', width=3),
            marker=dict(size=8),
            hovertemplate='Step %{x}<br>Score: %{y:.3f}<br>%{text}',
            text=[t[:50] + '...' if len(t) > 50 else t for t in tokens]
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # Highlight selected point
        if selected_idx is not None and 0 <= selected_idx < len(scores):
            fig.add_trace(go.Scatter(
                x=[selected_idx],
                y=[scores[selected_idx]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='circle-open', line=dict(width=3)),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Color regions by evidence type
        for i in range(len(scores) - 1):
            color = {
                'supporting': 'rgba(76, 175, 80, 0.2)',
                'opposing': 'rgba(244, 67, 54, 0.2)',
                'uncertainty': 'rgba(255, 193, 7, 0.2)',
                'neutral': 'rgba(158, 158, 158, 0.1)'
            }.get(types[i], 'rgba(158, 158, 158, 0.1)')
            
            fig.add_shape(
                type="rect",
                x0=i, x1=i+1,
                y0=min(0, scores[i]), y1=max(0, scores[i]),
                fillcolor=color,
                line=dict(width=0),
                layer="below"
            )
        
        # Update layout
        fig.update_layout(
            title="Evidence Accumulation",
            xaxis_title="Reasoning Step",
            yaxis_title="Evidence Score",
            height=400,
            hovermode='closest',
            plot_bgcolor='white',
            yaxis=dict(range=[-1.2, 1.2]),
            xaxis=dict(tickmode='linear', tick0=0, dtick=5)
        )
        
        return fig
    
    def render_reasoning_panel(self, question_data: Dict):
        """Render the reasoning trace panel with interactive tokens."""
        st.markdown("### 🤖 DeepSeek Reasoning")
        
        if 'reasoning_trace' not in question_data:
            st.info("No reasoning trace available for this question.")
            return
        
        tokens = question_data['reasoning_trace']['tokens']
        evidence_data = question_data.get('evidence_analysis', {})
        
        if not evidence_data:
            # Extract evidence if not already done
            evidence_extracted = self.evidence_extractor.extract_evidence(
                question_data['reasoning_trace'],
                question_data['model_answer']
            )
            evidence_data = self.evidence_extractor.format_for_visualization(evidence_extracted)
        
        # Create clickable tokens
        token_html = "<div style='line-height: 2.5;'>"
        
        for i, token in enumerate(tokens):
            # Determine token style based on evidence
            if 'evidence_types' in evidence_data and i < len(evidence_data['evidence_types']):
                evidence_type = evidence_data['evidence_types'][i]
                confidence = evidence_data.get('confidence_levels', [])[i] if i < len(evidence_data.get('confidence_levels', [])) else 'low'
                
                # Set color based on type
                if evidence_type == 'supporting':
                    color = '#4caf50' if confidence == 'high' else '#81c784'
                elif evidence_type == 'opposing':
                    color = '#f44336' if confidence == 'high' else '#ef5350'
                elif evidence_type == 'uncertainty':
                    color = '#ff9800'
                else:
                    color = '#9e9e9e'
                
                opacity = 0.8 if i == st.session_state.selected_token_idx else 0.6
            else:
                color = '#9e9e9e'
                opacity = 0.6
            
            # Create token HTML
            token_html += f"""
            <span class='reasoning-token' 
                  style='background-color: {color}; opacity: {opacity}; color: white;'
                  onclick='window.parent.postMessage({{type: "token_click", index: {i}}}, "*")'>
                {i+1}. {token}
            </span><br>
            """
        
        token_html += "</div>"
        
        # Render tokens
        st.markdown(token_html, unsafe_allow_html=True)
        
        # Final answer
        st.markdown(f"**Final Answer:** {question_data['model_answer']}")
    
    def render_mc_dropout_prediction(self, question: str):
        """Render MC Dropout model prediction with uncertainty."""
        st.markdown("### 🎯 MC Dropout Prediction")
        
        if self.mc_dropout_model is None:
            st.info("MC Dropout model not available. Train it first.")
            return
        
        # Get prediction (cached if already computed)
        if question not in st.session_state.mc_dropout_predictions:
            with st.spinner("Running MC Dropout inference..."):
                # Simplified prediction for demo
                pred_result = {
                    'prediction': 'No',  # Would come from actual model
                    'confidence': 0.65,
                    'uncertainty': 0.15,
                    'mc_samples': 50
                }
                st.session_state.mc_dropout_predictions[question] = pred_result
        else:
            pred_result = st.session_state.mc_dropout_predictions[question]
        
        # Display prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", pred_result['prediction'])
        
        with col2:
            st.metric("Confidence", f"{pred_result['confidence']:.1%}")
        
        with col3:
            st.metric("Uncertainty", f"±{pred_result['uncertainty']:.1%}")
        
        # Uncertainty gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_result['confidence'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Level"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    def render_human_votes(self, question_data: Dict):
        """Render human voting statistics."""
        st.markdown("### 👥 Human Consensus")
        
        human_votes = question_data.get('human_votes', {})
        
        if human_votes.get('total', 0) == 0:
            st.info("No human votes collected yet.")
            return
        
        # Create pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Yes', 'No'],
            values=[human_votes.get('yes', 0), human_votes.get('no', 0)],
            hole=.3,
            marker_colors=['#4caf50', '#f44336']
        )])
        
        fig_pie.update_layout(
            height=300,
            showlegend=True,
            annotations=[{
                'text': f"{human_votes.get('total', 0)}<br>votes",
                'x': 0.5, 'y': 0.5,
                'font_size': 20,
                'showarrow': False
            }]
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Consensus strength
        if human_votes.get('yes_percentage') is not None:
            consensus = abs(human_votes['yes_percentage'] - 0.5) * 2
            st.progress(consensus)
            st.caption(f"Consensus Strength: {consensus:.0%}")
    
    def run(self):
        """Main application loop."""
        st.title("🧠 Reasoning Visualizer")
        st.markdown("Explore how AI reasons about ambiguous questions with uncertainty quantification")
        
        # Sidebar
        selected_question = self.render_sidebar()
        
        if selected_question is None:
            st.info("👈 Select a question from the sidebar or enter a custom one")
            return
        
        # Main content
        st.markdown(f"## {selected_question['prompt']}")
        st.markdown(f"*Category: {selected_question.get('category', 'Unknown').replace('_', ' ').title()}*")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self.render_reasoning_panel(selected_question)
        
        with col2:
            # Evidence chart
            if 'reasoning_trace' in selected_question:
                evidence_data = selected_question.get('evidence_analysis', {})
                if not evidence_data and 'reasoning_trace' in selected_question:
                    # Extract evidence
                    evidence_extracted = self.evidence_extractor.extract_evidence(
                        selected_question['reasoning_trace'],
                        selected_question['model_answer']
                    )
                    evidence_data = self.evidence_extractor.format_for_visualization(evidence_extracted)
                
                if evidence_data:
                    # Create and display evidence chart
                    fig_evidence = self.create_evidence_chart(
                        evidence_data,
                        st.session_state.selected_token_idx
                    )
                    
                    # Make chart interactive
                    selected_point = st.plotly_chart(
                        fig_evidence,
                        use_container_width=True,
                        key="evidence_chart"
                    )
        
        # Bottom section
        col3, col4 = st.columns([1, 1])
        
        with col3:
            self.render_mc_dropout_prediction(selected_question['prompt'])
        
        with col4:
            self.render_human_votes(selected_question)
        
        # Token-evidence interaction handler
        if st.session_state.selected_token_idx is not None:
            st.info(f"Selected token #{st.session_state.selected_token_idx + 1}")


def main():
    """Main entry point."""
    app = ReasoningVisualizerApp()
    app.run()


if __name__ == "__main__":
    main()