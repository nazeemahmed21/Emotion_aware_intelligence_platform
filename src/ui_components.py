"""
Modern UI components for the Emotion-Aware Voice Feedback Bot
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

class ModernUI:
    """Modern UI component library for the voice bot"""
    
    # Color palette
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'success': '#28A745',
        'warning': '#FFC107',
        'danger': '#DC3545',
        'info': '#17A2B8',
        'light': '#F8F9FA',
        'dark': '#343A40',
        'muted': '#6C757D',
        'white': '#FFFFFF',
        'gradient_start': '#667eea',
        'gradient_end': '#764ba2'
    }
    
    # Emotion color mapping
    EMOTION_COLORS = {
        'happy': '#28A745',
        'excited': '#FFC107', 
        'calm': '#17A2B8',
        'neutral': '#6C757D',
        'sad': '#6F42C1',
        'angry': '#DC3545',
        'fearful': '#FD7E14',
        'surprised': '#E83E8C'
    }
    
    # Emotion emojis
    EMOTION_EMOJIS = {
        'happy': '😊',
        'excited': '🤩',
        'calm': '😌', 
        'neutral': '😐',
        'sad': '😢',
        'angry': '😠',
        'fearful': '😰',
        'surprised': '😲'
    }
    
    @staticmethod
    def inject_custom_css():
        """Inject modern CSS styling"""
        st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .main {
            font-family: 'Inter', sans-serif;
        }
        
        /* Header Styles */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        
        /* Card Styles */
        .modern-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid #E9ECEF;
            margin: 1rem 0;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .modern-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }
        
        .card-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #2E86AB;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Recording Button Styles */
        .record-button {
            background: linear-gradient(135deg, #DC3545 0%, #C82333 100%);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(220, 53, 69, 0.3);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 1rem auto;
        }
        
        .record-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(220, 53, 69, 0.4);
        }
        
        .record-button.recording {
            background: linear-gradient(135deg, #28A745 0%, #20C997 100%);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3); }
            50% { box-shadow: 0 4px 25px rgba(40, 167, 69, 0.6); }
            100% { box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3); }
        }
        
        /* Emotion Display */
        .emotion-display {
            text-align: center;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
        }
        
        .emotion-emoji {
            font-size: 4rem;
            margin-bottom: 0.5rem;
        }
        
        .emotion-name {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .emotion-confidence {
            font-size: 1rem;
            opacity: 0.8;
        }
        
        /* Response Bubble */
        .response-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 20px;
            margin: 1rem 0;
            position: relative;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }
        
        .response-bubble::before {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 30px;
            width: 0;
            height: 0;
            border-left: 15px solid transparent;
            border-right: 15px solid transparent;
            border-top: 15px solid #667eea;
        }
        
        .response-text {
            font-size: 1.1rem;
            line-height: 1.6;
            margin: 0;
        }
        
        /* Transcription Display */
        .transcription-display {
            background: #F8F9FA;
            border-left: 4px solid #2E86AB;
            padding: 1rem 1.5rem;
            border-radius: 0 10px 10px 0;
            font-style: italic;
            margin: 1rem 0;
        }
        
        /* Status Indicators */
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        .status-success {
            background: #D4EDDA;
            color: #155724;
            border: 1px solid #C3E6CB;
        }
        
        .status-warning {
            background: #FFF3CD;
            color: #856404;
            border: 1px solid #FFEAA7;
        }
        
        .status-error {
            background: #F8D7DA;
            color: #721C24;
            border: 1px solid #F5C6CB;
        }
        
        .status-info {
            background: #D1ECF1;
            color: #0C5460;
            border: 1px solid #BEE5EB;
        }
        
        /* Progress Bars */
        .progress-container {
            background: #E9ECEF;
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        
        /* Conversation History */
        .conversation-item {
            background: white;
            border-radius: 15px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #2E86AB;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .conversation-meta {
            font-size: 0.8rem;
            color: #6C757D;
            margin-bottom: 0.5rem;
        }
        
        /* Sidebar Styles */
        .sidebar-section {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .sidebar-header {
            font-weight: 600;
            color: #2E86AB;
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }
        
        /* Audio Visualizer */
        .audio-visualizer {
            background: #F8F9FA;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            text-align: center;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            
            .modern-card {
                padding: 1rem;
            }
            
            .emotion-emoji {
                font-size: 3rem;
            }
        }
        
        /* Hide Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #F1F1F1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #C1C1C1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #A8A8A8;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render modern header"""
        st.markdown("""
        <div class="main-header">
            <h1>🎭 Emotion-Aware Voice Bot</h1>
            <p>Advanced AI-powered emotional intelligence for natural conversations</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_card(title: str, content: str, icon: str = "📋"):
        """Render a modern card component"""
        st.markdown(f"""
        <div class="modern-card">
            <div class="card-header">
                <span>{icon}</span>
                <span>{title}</span>
            </div>
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_emotion_display(emotion: str, confidence: float):
        """Render emotion display with modern styling"""
        emoji = ModernUI.EMOTION_EMOJIS.get(emotion, '😐')
        color = ModernUI.EMOTION_COLORS.get(emotion, ModernUI.COLORS['muted'])
        
        st.markdown(f"""
        <div class="emotion-display" style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); border: 2px solid {color}40;">
            <div class="emotion-emoji">{emoji}</div>
            <div class="emotion-name" style="color: {color};">{emotion.capitalize()}</div>
            <div class="emotion-confidence">Confidence: {confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_response_bubble(response: str):
        """Render AI response in a modern bubble"""
        st.markdown(f"""
        <div class="response-bubble">
            <p class="response-text">{response}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_transcription(text: str):
        """Render transcription with modern styling"""
        st.markdown(f"""
        <div class="transcription-display">
            <strong>🗣️ You said:</strong> "{text}"
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_status_indicator(status: str, message: str):
        """Render status indicator"""
        status_class = f"status-{status}"
        icons = {
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'info': 'ℹ️'
        }
        icon = icons.get(status, 'ℹ️')
        
        st.markdown(f"""
        <div class="status-indicator {status_class}">
            <span>{icon}</span>
            <span>{message}</span>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_progress_bar(progress: float, label: str = ""):
        """Render modern progress bar"""
        st.markdown(f"""
        <div>
            {f'<div style="margin-bottom: 0.5rem; font-weight: 500;">{label}</div>' if label else ''}
            <div class="progress-container">
                <div class="progress-bar" style="width: {progress * 100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_emotion_gauge(emotion: str, confidence: float):
        """Create a modern emotion confidence gauge"""
        color = ModernUI.EMOTION_COLORS.get(emotion, ModernUI.COLORS['muted'])
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{emotion.capitalize()} Confidence", 'font': {'size': 16, 'family': 'Inter'}},
            delta = {'reference': 70, 'increasing': {'color': color}},
            gauge = {
                'axis': {'range': [None, 100], 'tickfont': {'size': 12, 'family': 'Inter'}},
                'bar': {'color': color, 'thickness': 0.8},
                'steps': [
                    {'range': [0, 50], 'color': "#F8F9FA"},
                    {'range': [50, 80], 'color': "#E9ECEF"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                },
                'bgcolor': 'white',
                'borderwidth': 2,
                'bordercolor': color
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            font={'family': 'Inter'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    @staticmethod
    def create_waveform_plot(audio_data: np.ndarray, sample_rate: int = 16000):
        """Create modern audio waveform visualization"""
        time_axis = np.arange(len(audio_data)) / sample_rate
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=audio_data,
            mode='lines',
            name='Waveform',
            line=dict(
                color='rgba(102, 126, 234, 0.8)',
                width=1.5
            ),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        
        fig.update_layout(
            title={
                'text': '🎵 Audio Waveform',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Inter'}
            },
            xaxis_title='Time (seconds)',
            yaxis_title='Amplitude',
            height=300,
            margin=dict(l=40, r=40, t=60, b=40),
            font={'family': 'Inter'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            )
        )
        
        return fig
    
    @staticmethod
    def create_emotion_history_chart(conversation_history: List[Dict]):
        """Create emotion history chart"""
        if not conversation_history:
            return None
        
        df = pd.DataFrame(conversation_history)
        
        # Count emotions
        emotion_counts = df['emotion'].value_counts()
        
        colors = [ModernUI.EMOTION_COLORS.get(emotion, ModernUI.COLORS['muted']) 
                 for emotion in emotion_counts.index]
        
        fig = go.Figure(data=[
            go.Bar(
                x=emotion_counts.index,
                y=emotion_counts.values,
                marker_color=colors,
                text=emotion_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title={
                'text': '📊 Emotion Distribution',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Inter'}
            },
            xaxis_title='Emotions',
            yaxis_title='Count',
            height=300,
            margin=dict(l=40, r=40, t=60, b=40),
            font={'family': 'Inter'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        return fig