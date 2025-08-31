#!/usr/bin/env python3
"""
Emotion-Aware Voice Intelligence Platform
=========================================

Enterprise-grade emotion analysis powered by Hume AI

This application provides comprehensive emotional analysis of voice recordings using
Hume AI's state-of-the-art emotion recognition models. It features both voice recording
and file upload capabilities with professional visualizations and insights.

Features:
- Real-time voice recording with audio-recorder-streamlit
- File upload support for multiple audio formats
- Multi-model emotion analysis (Prosody, Burst, Language)
- Professional visualizations and charts
- Hesitancy and uncertainty pattern detection
- Enterprise-ready error handling and logging
- Responsive design with modern UI/UX

Author: Emotion AI Team
Version: 2.0.0
License: MIT
"""

# Standard library imports
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/emotion_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add knowledge_base to path for Hume imports
knowledge_base_path = Path(__file__).parent / "knowledge_base"
sys.path.append(str(knowledge_base_path))

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Application Configuration
class Config:
    """Application configuration management"""
    
    # Page settings
    PAGE_TITLE = "Emotion-Aware Voice Intelligence Platform"
    PAGE_ICON = "🧠"
    LAYOUT = "wide"
    
    # Audio settings
    SUPPORTED_FORMATS = ['wav', 'mp3', 'm4a', 'ogg', 'flac']
    MAX_FILE_SIZE_MB = 200
    SAMPLE_RATE = 44100
    PAUSE_THRESHOLD = 2.0
    
    # Analysis settings
    ANALYSIS_DEPTHS = ["Standard", "Comprehensive", "Research Grade"]
    DEFAULT_ANALYSIS_DEPTH = "Comprehensive"
    
    # UI settings
    RECORDING_COLOR = "#ff6b6b"
    NEUTRAL_COLOR = "#667eea"
    
    @staticmethod
    def validate_environment() -> bool:
        """Validate required environment variables"""
        required_vars = ['HUME_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False
        return True

# Initialize configuration
config = Config()

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
        margin: 1rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 500;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 500;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9ff 0%, #ffffff 100%);
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border-radius: 10px;
        border: 1px solid #e1e8ed;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #e1e8ed;
        margin-top: 3rem;
    }
    
         /* Voice Recorder Styling */
     .stTabs [data-baseweb="tab-list"] {
         gap: 2rem;
         background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
         border-radius: 15px;
         padding: 0.5rem;
     }
     
     .stTabs [data-baseweb="tab"] {
         background: transparent;
         border-radius: 10px;
         padding: 0.75rem 1.5rem;
         font-weight: 600;
         transition: all 0.3s ease;
         color: #666 !important;
         border: 1px solid transparent;
     }
     
     .stTabs [data-baseweb="tab"]:hover {
         background: rgba(102, 126, 234, 0.1);
         color: #667eea !important;
         border-color: rgba(102, 126, 234, 0.3);
     }
     
     .stTabs [aria-selected="true"] {
         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
         color: white !important;
         box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
         border-color: #667eea;
     }
    
    /* Audio recorder container */
    .stAudio {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
        margin: 1rem 0;
    }
    
    /* Recording instructions */
    .recording-instructions {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Success message styling */
    .element-container .stSuccess {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0fff0 100%);
        border: 1px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🧠 Emotion-Aware Voice Intelligence Platform</div>
    <div class="main-subtitle">Advanced AI-Powered Emotional Analysis & Insights</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # System status
    st.markdown("#### System Status")
    
    # Check API key
    api_key = os.getenv('HUME_API_KEY')
    if api_key:
        st.markdown('<div class="status-success">✅ Hume AI Connected</div>', unsafe_allow_html=True)
        st.caption(f"API Key: {api_key[:10]}...")
    else:
        st.markdown('<div class="status-error">❌ API Key Missing</div>', unsafe_allow_html=True)
        st.error("Please add HUME_API_KEY to your .env file")
        st.stop()
    
    # Import check
    try:
        from knowledge_base.hume.hume_client import HumeClient, HumeConfig, GranularityLevel
        st.markdown('<div class="status-success">✅ Hume Client Ready</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown('<div class="status-error">❌ Import Error</div>', unsafe_allow_html=True)
        st.error(f"Cannot import Hume client: {e}")
        st.stop()
    
    st.markdown("---")
    
    # Analysis settings
    st.markdown("#### Analysis Settings")
    
    analysis_depth = st.selectbox(
        "Analysis Depth",
        ["Standard", "Comprehensive", "Research Grade"],
        index=1,
        help="Choose the depth of emotional analysis"
    )
    
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_model_breakdown = st.checkbox("Show Model Attribution", value=True)
    show_raw_data = st.checkbox("Include Raw Data", value=False)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("#### Session Info")
    st.info(f"**Session Started:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Help section
    with st.expander("📚 Quick Help"):
        st.markdown("""
        **Supported Formats:**
        - WAV, MP3, M4A, OGG, FLAC
        
        **Best Practices:**
        - Clear speech audio
        - 2-30 seconds duration
        - Minimal background noise
        
        **Analysis Models:**
        - 🎵 Prosody: Voice tone & rhythm
        - ⚡ Burst: Quick expressions
        - 📝 Language: Speech content
        """)

# Main content area
def analyze_with_hume(audio_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Analyze audio with Hume AI - Enterprise version with comprehensive error handling
    
    Args:
        audio_file_path (str): Path to the audio file to analyze
        
    Returns:
        Optional[Dict[str, Any]]: Analysis results or None if failed
        
    Raises:
        Exception: Re-raises exceptions after logging for upstream handling
    """
    try:
        logger.info(f"Starting Hume AI analysis for file: {audio_file_path}")
        
        # Validate file exists and is readable
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        if os.path.getsize(audio_file_path) == 0:
            raise ValueError("Audio file is empty")
        
        # Configure Hume client
        hume_config = HumeConfig(
            api_key=api_key,
            secret_key=os.getenv('HUME_SECRET_KEY'),
            webhook_url=os.getenv('HUME_WEBHOOK_URL')
        )
        client = HumeClient(hume_config)
        granularity = GranularityLevel.UTTERANCE
        
        async def run_analysis() -> Optional[Dict[str, Any]]:
            """Async analysis execution with progress tracking"""
            try:
                # Submit job
                job_id = await client.submit_files([audio_file_path], granularity)
                logger.info(f"Hume AI job submitted with ID: {job_id}")
                
                # Progress tracking UI
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(f"🚀 Job submitted: {job_id}")
                progress_bar.progress(25)
                
                status_text.text("🔄 Processing with Hume AI...")
                progress_bar.progress(50)
                
                # Wait for completion
                success = await client.wait_for_job(job_id)
                progress_bar.progress(75)
                
                if success:
                    status_text.text("📥 Downloading results...")
                    predictions = await client.get_job_predictions(job_id, format="json")
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Clean up UI elements
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    logger.info("Hume AI analysis completed successfully")
                    return predictions
                else:
                    logger.error(f"Hume AI job {job_id} failed")
                    return None
                    
            except Exception as e:
                logger.error(f"Error in async analysis: {str(e)}")
                raise
        
        return asyncio.run(run_analysis())
        
    except Exception as e:
        error_msg = f"Hume AI analysis failed: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        return None

def extract_emotions(predictions: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract emotions from Hume predictions with comprehensive error handling
    
    Args:
        predictions: Raw predictions from Hume AI API
        
    Returns:
        List[Dict[str, Any]]: List of emotion dictionaries with name, score, model, and timestamp
    """
    if not predictions:
        logger.warning("No predictions provided to extract_emotions")
        return []
    
    all_emotions = []
    
    try:
        logger.info("Starting emotion extraction from Hume predictions")
        
        # Normalize predictions to list format
        predictions_list = predictions if isinstance(predictions, list) else [predictions]
        
        for file_idx, file_prediction in enumerate(predictions_list):
            logger.debug(f"Processing file prediction {file_idx + 1}/{len(predictions_list)}")
            
            # Handle different prediction structures
            if "results" in file_prediction:
                file_predictions = file_prediction["results"].get("predictions", [])
            else:
                file_predictions = [file_prediction]
            
            for pred_idx, pred in enumerate(file_predictions):
                models = pred.get("models", {})
                logger.debug(f"Found models: {list(models.keys())}")
                
                # Process each model (prosody, burst, language)
                for model_name in ["prosody", "burst", "language"]:
                    if model_name not in models:
                        logger.debug(f"Model {model_name} not found in predictions")
                        continue
                    
                    model_data = models[model_name]
                    grouped_predictions = model_data.get("grouped_predictions", [])
                    
                    for group_idx, group in enumerate(grouped_predictions):
                        predictions_inner = group.get("predictions", [])
                        
                        for inner_pred in predictions_inner:
                            if "emotions" not in inner_pred:
                                continue
                                
                            emotions = inner_pred["emotions"]
                            
                            for emotion in emotions:
                                emotion_data = {
                                    'name': emotion.get('name', 'unknown'),
                                    'score': float(emotion.get('score', 0.0)),
                                    'model': model_name,
                                    'timestamp': inner_pred.get('time', {}),
                                    'file_index': file_idx,
                                    'prediction_index': pred_idx,
                                    'group_index': group_idx
                                }
                                all_emotions.append(emotion_data)
        
        logger.info(f"Successfully extracted {len(all_emotions)} emotions from predictions")
        return all_emotions
        
    except Exception as e:
        error_msg = f"Error extracting emotions: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        return []

def create_emotion_visualizations(emotions, emotion_stats):
    """Create professional visualizations"""
    
    # Top 10 emotions bar chart
    top_10 = sorted(emotion_stats.items(), key=lambda x: x[1]['mean_score'], reverse=True)[:10]
    
    fig_bar = go.Figure(data=[
        go.Bar(
            x=[stats['mean_score'] for _, stats in top_10],
            y=[name for name, _ in top_10],
            orientation='h',
            marker=dict(
                color=[stats['mean_score'] for _, stats in top_10],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Intensity")
            ),
            text=[f"{stats['mean_score']:.3f}" for _, stats in top_10],
            textposition='auto',
        )
    ])
    
    fig_bar.update_layout(
        title="Top 10 Emotions by Intensity",
        xaxis_title="Mean Score",
        yaxis_title="Emotion",
        height=500,
        template="plotly_white",
        font=dict(family="Inter, sans-serif")
    )
    
    # Model distribution pie chart
    model_counts = {'prosody': 0, 'burst': 0, 'language': 0}
    for emotion in emotions:
        model_counts[emotion['model']] += 1
    
    fig_pie = go.Figure(data=[
        go.Pie(
            labels=['🎵 Prosody', '⚡ Burst', '📝 Language'],
            values=list(model_counts.values()),
            hole=0.4,
            marker=dict(colors=['#667eea', '#764ba2', '#f093fb'])
        )
    ])
    
    fig_pie.update_layout(
        title="Emotion Detection by AI Model",
        template="plotly_white",
        font=dict(family="Inter, sans-serif")
    )
    
    return fig_bar, fig_pie

def display_professional_results(emotions):
    """Display results with professional UI"""
    if not emotions:
        st.warning("⚠️ No emotions detected in the audio")
        st.info("This might occur with:")
        st.markdown("""
        - Very quiet or unclear audio
        - Non-speech content (music, noise)
        - Emotionally neutral speech
        - Technical audio issues
        """)
        return
    
    # Summary metrics
    st.markdown("## 📊 Analysis Overview")
    
    # Calculate stats
    emotion_stats = {}
    for emotion in emotions:
        name = emotion['name']
        score = emotion['score']
        model = emotion['model']
        
        if name not in emotion_stats:
            emotion_stats[name] = {
                'scores': [],
                'models': [],
                'total_score': 0,
                'count': 0
            }
        
        emotion_stats[name]['scores'].append(score)
        emotion_stats[name]['models'].append(model)
        emotion_stats[name]['total_score'] += score
        emotion_stats[name]['count'] += 1
    
    # Calculate final stats
    for name, stats in emotion_stats.items():
        stats['mean_score'] = stats['total_score'] / stats['count']
        stats['max_score'] = max(stats['scores'])
        stats['unique_models'] = list(set(stats['models']))
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>🎯 Total Emotions</h3>
            <h2 style="color: #667eea;">{}</h2>
            <p>Unique emotions detected</p>
        </div>
        """.format(len(emotion_stats)), unsafe_allow_html=True)
    
    with col2:
        dominant = max(emotion_stats.items(), key=lambda x: x[1]['mean_score'])
        st.markdown("""
        <div class="metric-container">
            <h3>👑 Dominant Emotion</h3>
            <h2 style="color: #764ba2;">{}</h2>
            <p>Score: {:.3f}</p>
        </div>
        """.format(dominant[0], dominant[1]['mean_score']), unsafe_allow_html=True)
    
    with col3:
        avg_intensity = np.mean([e['score'] for e in emotions])
        st.markdown("""
        <div class="metric-container">
            <h3>⚡ Avg Intensity</h3>
            <h2 style="color: #f093fb;">{:.3f}</h2>
            <p>Overall emotional intensity</p>
        </div>
        """.format(avg_intensity), unsafe_allow_html=True)
    
    with col4:
        total_detections = len(emotions)
        st.markdown("""
        <div class="metric-container">
            <h3>🔍 Total Detections</h3>
            <h2 style="color: #667eea;">{}</h2>
            <p>Across all models</p>
        </div>
        """.format(total_detections), unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("## 📈 Emotional Insights")
    
    fig_bar, fig_pie = create_emotion_visualizations(emotions, emotion_stats)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed results table
    st.markdown("## 🏆 Top 10 Emotions Ranking")
    
    top_10 = sorted(emotion_stats.items(), key=lambda x: x[1]['mean_score'], reverse=True)[:10]
    
    table_data = []
    for i, (name, stats) in enumerate(top_10):
        # Intensity indicator
        if stats['mean_score'] > 0.7:
            intensity = "🔥 High"
        elif stats['mean_score'] > 0.4:
            intensity = "🔸 Medium"
        else:
            intensity = "🔹 Low"
        
        table_data.append({
            'Rank': f"#{i+1}",
            'Emotion': name,
            'Mean Score': f"{stats['mean_score']:.3f}",
            'Peak Score': f"{stats['max_score']:.3f}",
            'Frequency': stats['count'],
            'Intensity': intensity,
            'Models': ', '.join(stats['unique_models'])
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Hesitancy analysis
    st.markdown("## 🤔 Hesitancy & Uncertainty Analysis")
    
    hesitancy_keywords = [
        'anxiety', 'nervousness', 'uncertainty', 'confusion', 
        'hesitation', 'doubt', 'worry', 'stress', 'awkwardness'
    ]
    
    hesitancy_emotions = []
    for name, stats in emotion_stats.items():
        if any(keyword in name.lower() for keyword in hesitancy_keywords):
            hesitancy_emotions.append((name, stats['mean_score'], stats['count']))
    
    if hesitancy_emotions:
        hesitancy_emotions.sort(key=lambda x: x[1], reverse=True)
        avg_hesitancy = sum(score for _, score, _ in hesitancy_emotions) / len(hesitancy_emotions)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Hesitancy", f"{avg_hesitancy:.3f}")
        with col2:
            st.metric("Hesitancy Indicators", len(hesitancy_emotions))
        with col3:
            level = "High" if avg_hesitancy > 0.5 else "Medium" if avg_hesitancy > 0.3 else "Low"
            st.metric("Hesitancy Level", level)
        
        # Hesitancy details
        hesitancy_data = []
        for name, score, count in hesitancy_emotions:
            hesitancy_data.append({
                'Emotion': name,
                'Score': f"{score:.3f}",
                'Frequency': count,
                'Impact': "High" if score > 0.5 else "Medium" if score > 0.3 else "Low"
            })
        
        if hesitancy_data:
            st.markdown("### Detected Hesitancy Patterns")
            hesitancy_df = pd.DataFrame(hesitancy_data)
            st.dataframe(hesitancy_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No significant hesitancy patterns detected")
        st.info("The speaker appears confident and decisive in their communication.")
    
    # Model breakdown
    if show_model_breakdown:
        st.markdown("## 🧠 AI Model Performance Analysis")
        
        model_stats = {'prosody': [], 'burst': [], 'language': []}
        for emotion in emotions:
            model_stats[emotion['model']].append(emotion['score'])
        
        col1, col2, col3 = st.columns(3)
        
        models_info = {
            'prosody': {
                'name': '🎵 Prosody Model',
                'description': 'Analyzes vocal tone, pitch, rhythm, and speech patterns',
                'color': '#667eea'
            },
            'burst': {
                'name': '⚡ Burst Model', 
                'description': 'Detects quick emotional expressions and vocal bursts',
                'color': '#764ba2'
            },
            'language': {
                'name': '📝 Language Model',
                'description': 'Processes emotional content from speech transcription',
                'color': '#f093fb'
            }
        }
        
        for i, (model_key, scores) in enumerate(model_stats.items()):
            model_info = models_info[model_key]
            
            with [col1, col2, col3][i]:
                if scores:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: {model_info['color']};">{model_info['name']}</h3>
                        <p style="font-size: 0.9rem; color: #666;">{model_info['description']}</p>
                        <div style="margin: 1rem 0;">
                            <strong>Emotions Detected:</strong> {len(scores)}<br>
                            <strong>Average Score:</strong> {np.mean(scores):.3f}<br>
                            <strong>Peak Score:</strong> {max(scores):.3f}<br>
                            <strong>Contribution:</strong> {len(scores)/len(emotions)*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: {model_info['color']};">{model_info['name']}</h3>
                        <p style="font-size: 0.9rem; color: #666;">{model_info['description']}</p>
                        <p style="color: #999;">No emotions detected by this model</p>
                    </div>
                    """, unsafe_allow_html=True)

# Main audio input section
st.markdown("## 🎤 Audio Analysis")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["🎙️ Record Voice", "📁 Upload File", "🎯 Interview Practice"])

with tab1:
    st.markdown("### 🎙️ Record Your Voice")
    
    # Recording instructions
    st.markdown("""
    <div class="recording-instructions">
        <h4 style="color: #667eea; margin-bottom: 1rem;">📋 Recording Tips for Best Results:</h4>
        <ul style="color: #333; line-height: 1.6;">
            <li><strong style="color: #764ba2;">🔇 Quiet Environment:</strong> Find a quiet space with minimal background noise</li>
            <li><strong style="color: #764ba2;">🎤 Clear Speech:</strong> Speak clearly and naturally, as you would in conversation</li>
            <li><strong style="color: #764ba2;">⏱️ Duration:</strong> Record for 10-30 seconds for optimal emotion detection</li>
            <li><strong style="color: #764ba2;">😊 Be Expressive:</strong> Let your natural emotions come through in your voice</li>
            <li><strong style="color: #764ba2;">📱 Device:</strong> Use a good quality microphone if available</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Import audio recorder
    from audio_recorder_streamlit import audio_recorder
    
    # Voice recorder with custom styling
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#ff6b6b",
        neutral_color="#667eea",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=2.0,
        sample_rate=44100
    )
    
    # Handle recorded audio
    if audio_bytes:
        st.success("✅ Recording captured successfully!")
        
        # Display audio player
        st.audio(audio_bytes, format='audio/wav')
        
        # Analysis button for recorded audio
        if st.button("🚀 Analyze Recorded Voice", type="primary", use_container_width=True):
            with st.spinner(""):
                try:
                    # Create temporary file from recorded audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(audio_bytes)
                        tmp_file.flush()
                        tmp_path = tmp_file.name
                    
                    time.sleep(0.1)  # Windows file handling
                    
                    # Analyze with Hume
                    predictions = analyze_with_hume(tmp_path)
                    
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except (PermissionError, FileNotFoundError):
                        pass
                    
                    if predictions:
                        # Extract and display emotions
                        emotions = extract_emotions(predictions)
                        display_professional_results(emotions)
                        
                        # Raw data section
                        if show_raw_data:
                            with st.expander("🔍 Raw Hume AI Response Data"):
                                st.json(predictions)
                    else:
                        st.error("❌ Analysis failed. Please try again or contact support.")
                
                except Exception as e:
                    st.error(f"❌ Error processing recorded audio: {e}")
                    with st.expander("🔧 Technical Details"):
                        import traceback
                        st.code(traceback.format_exc())

with tab2:
    st.markdown("### Upload Audio File")
    
    # Upload area
    uploaded_file = st.file_uploader(
        "Upload your audio file for emotional analysis",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
        help="Supported formats: WAV, MP3, M4A, OGG, FLAC. Best results with clear speech audio."
    )

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    # File info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.success(f"✅ **{uploaded_file.name}** uploaded successfully")
        st.caption(f"File size: {file_size_mb:.2f} MB")
    
    with col2:
        # Audio player
        st.audio(uploaded_file.read(), format='audio/wav')
        uploaded_file.seek(0)  # Reset file pointer
    
    # Analysis button
    if st.button("🚀 Start Emotional Analysis", type="primary", use_container_width=True):
        with st.spinner(""):
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file.flush()
                    tmp_path = tmp_file.name
                
                time.sleep(0.1)  # Windows file handling
                
                # Analyze with Hume
                predictions = analyze_with_hume(tmp_path)
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except (PermissionError, FileNotFoundError):
                    pass
                
                if predictions:
                    # Extract and display emotions
                    emotions = extract_emotions(predictions)
                    display_professional_results(emotions)
                    
                    # Raw data section
                    if show_raw_data:
                        with st.expander("🔍 Raw Hume AI Response Data"):
                            st.json(predictions)
                else:
                    st.error("❌ Analysis failed. Please try again or contact support.")
            
            except Exception as e:
                st.error(f"❌ Error processing audio: {e}")
                with st.expander("🔧 Technical Details"):
                    import traceback
                    st.code(traceback.format_exc())

else:
         # Welcome message
     st.markdown("""
     <div class="upload-area">
         <h3 style="color: #667eea; margin-bottom: 1rem;">🎤 Ready to Analyze Emotions</h3>
         <p style="color: #333; font-size: 1.1rem; margin-bottom: 0.5rem;">Upload an audio file to begin advanced emotional intelligence analysis</p>
         <p style="color: #666; font-size: 0.9rem; margin: 0;">
             Drag and drop your file above or click to browse
         </p>
     </div>
     """, unsafe_allow_html=True)

# Interview Practice Tab
with tab3:
    # Clean header with better spacing
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #667eea; margin-bottom: 0.5rem;">🎯 Interview Practice & Coaching</h1>
        <p style="color: #666; font-size: 1.1rem; margin: 0;">AI-powered interview preparation with personalized feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize question manager in session state
    if 'question_manager' not in st.session_state:
        try:
            from src.rag.question_manager import QuestionManager
            st.session_state.question_manager = QuestionManager()
            st.success("✅ Interview question system loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load interview questions: {e}")
            st.session_state.question_manager = None
    
    if st.session_state.question_manager:
        # Clean role selection section
        st.markdown("### 🎭 Role Selection")
        
        # Role selection in a clean card with better alignment
        st.markdown("""
        <div style="background: transparent; padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.2); box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 1rem 0;">
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([4, 1, 0.75])  # Made Get Question button smaller by reducing its column width
        
        with col1:
            available_roles = st.session_state.question_manager.get_available_file_types()
            selected_role = st.selectbox(
                "Choose your interview role:",
                options=available_roles,
                format_func=lambda x: x.replace('_', ' ').title(),
                help="Select the type of interview questions you want to practice"
            )
        
        with col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)  # Increased spacer to move button down more
            if st.button("🔄 Reset", help="Start fresh with new question order", use_container_width=True):
                st.session_state.question_manager.reset_session()
                st.session_state.current_question = None
                st.session_state.user_answer = ""
                st.rerun()
        
        with col3:
            if selected_role and st.button("🎲 Get Question", type="primary", use_container_width=True):
                question = st.session_state.question_manager.get_next_question()
                if question:
                    st.session_state.current_question = question
                    st.session_state.user_answer = ""
                    st.rerun()
                else:
                    st.error("❌ Failed to get question. Please try again.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Select role and get questions
        if selected_role and st.session_state.question_manager.select_file_type(selected_role):
            # Clean progress display
            progress = st.session_state.question_manager.get_progress()
            
            # Progress section with better visual hierarchy
            st.markdown("---")
            st.markdown("### 📊 Practice Progress")
            
            # Progress metrics in a clean grid
            progress_cols = st.columns(4)
            with progress_cols[0]:
                st.metric("Cycle", f"#{progress['current_cycle']}")
            with progress_cols[1]:
                st.metric("Progress", f"{progress['asked_questions']}/{progress['total_questions']}")
            with progress_cols[2]:
                st.metric("Remaining", progress['remaining_questions'])
            with progress_cols[3]:
                st.metric("Completion", f"{progress['progress_percentage']:.1f}%")
            
            # Clean progress bar
            st.progress(progress['progress_percentage'] / 100)
            
            # Display current question
            if 'current_question' in st.session_state and st.session_state.current_question:
                question = st.session_state.current_question
                
                # Clean question display
                st.markdown("---")
                st.markdown("### ❓ Current Question")
                
                # Question in a clean card
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 1rem 0;">
                    <h4 style="color: #333; margin-bottom: 1rem;">{question['question']}</h4>
                    <div style="display: flex; gap: 1rem; color: #666; font-size: 0.9rem;">
                        <span><strong>Category:</strong> {question['category']}</span>
                        <span><strong>ID:</strong> {question['id']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Answer input section
                st.markdown("### 💭 Your Answer")
                
                # Clean tabs for input methods
                answer_tab1, answer_tab2 = st.tabs(["📝 Text Answer", "🎤 Voice Answer"])
                
                with answer_tab1:
                    user_answer = st.text_area(
                        "Type your answer here...",
                        value=st.session_state.get('user_answer', ''),
                        height=150,
                        placeholder="Share your thoughts, experiences, or technical knowledge...",
                        help="Be specific and use the STAR method for behavioral questions"
                    )
                
                with answer_tab2:
                    # Clean voice recording section
                    st.markdown("#### 🎤 Record Your Voice Answer")
                    
                    # Recording tips in a clean card
                    with st.expander("📋 Recording Tips", expanded=False):
                        st.markdown("""
                        **For best results:**
                        - 🔇 Find a quiet environment
                        - 🎤 Speak clearly and naturally
                        - ⏱️ Record for 30 seconds to 2 minutes
                        - 😊 Let your emotions come through
                        - 💭 Share your thought process
                        """)
                    
                    # Voice recorder
                    voice_answer = audio_recorder(
                        text="Click to record your answer",
                        recording_color="#ff6b6b",
                        neutral_color="#667eea",
                        icon_name="microphone",
                        icon_size="2x",
                        pause_threshold=2.0,
                        sample_rate=44100
                    )
                    
                    if voice_answer:
                        st.success("✅ Voice answer recorded successfully!")
                        st.audio(voice_answer, format='audio/wav')
                        
                        # Store voice answer in session state
                        st.session_state.voice_answer = voice_answer
                        
                        # Clean action buttons
                        st.markdown("#### 🔄 Process Your Recording")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("📝 Transcribe Voice", type="secondary", use_container_width=True):
                                with st.spinner("Transcribing voice to text..."):
                                    try:
                                        from src.speech_to_text import WhisperTranscriber
                                        transcriber = WhisperTranscriber()
                                        transcription_result = transcriber.transcribe_audio(voice_answer)
                                        
                                        if transcription_result and 'text' in transcription_result:
                                            st.session_state.voice_transcription = transcription_result['text']
                                            st.session_state.transcription_confidence = transcription_result.get('confidence', 0.0)
                                            
                                            st.success("✅ Transcription completed!")
                                            st.info(f"**Transcribed Text:** {transcription_result['text']}")
                                            
                                            if 'confidence' in transcription_result:
                                                st.info(f"**Confidence:** {transcription_result['confidence']:.2f}")
                                        else:
                                            st.error("❌ Transcription failed. Please try again.")
                                            
                                    except Exception as e:
                                        st.error(f"❌ Transcription error: {e}")
                                        with st.expander("🔧 Technical Details"):
                                            import traceback
                                            st.code(traceback.format_exc())
                        
                        with col2:
                            if st.button("🧠 Analyze Emotions", type="secondary", use_container_width=True):
                                with st.spinner("Analyzing voice emotions..."):
                                    try:
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                            tmp_file.write(voice_answer)
                                            tmp_file.flush()
                                            tmp_path = tmp_file.name
                                        
                                        time.sleep(0.1)
                                        predictions = analyze_with_hume(tmp_path)
                                        
                                        try:
                                            os.unlink(tmp_path)
                                        except (PermissionError, FileNotFoundError):
                                            pass
                                        
                                        if predictions:
                                            emotions = extract_emotions(predictions)
                                            st.session_state.voice_emotions = emotions
                                            
                                            # Clean emotion display
                                            st.markdown("#### 🎵 Voice Emotion Analysis")
                                            
                                            if emotions:
                                                # Top 5 emotions in clean format
                                                emotion_stats = {}
                                                for emotion in emotions:
                                                    name = emotion['name']
                                                    score = emotion['score']
                                                    if name not in emotion_stats:
                                                        emotion_stats[name] = []
                                                    emotion_stats[name].append(score)
                                                
                                                for name, scores in emotion_stats.items():
                                                    emotion_stats[name] = np.mean(scores)
                                                
                                                top_5 = sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True)[:5]
                                                
                                                # Clean emotion metrics
                                                emotion_cols = st.columns(5)
                                                for i, (emotion, score) in enumerate(top_5):
                                                    with emotion_cols[i]:
                                                        st.metric(f"#{i+1}", f"{score:.3f}", emotion.title())
                                                
                                                # Dominant emotion insight
                                                dominant_emotion = top_5[0] if top_5 else None
                                                if dominant_emotion:
                                                    st.info(f"🎯 **Dominant Emotion:** {dominant_emotion[0].title()} (Score: {dominant_emotion[1]:.3f})")
                                                    
                                                    # Simple coaching based on dominant emotion
                                                    if 'confidence' in dominant_emotion[0].lower() or 'enthusiasm' in dominant_emotion[0].lower():
                                                        st.success("🌟 Great energy! Your voice conveys confidence and enthusiasm.")
                                                    elif 'nervousness' in dominant_emotion[0].lower() or 'anxiety' in dominant_emotion[0].lower():
                                                        st.warning("😌 Take a breath! Try to slow down and speak more deliberately.")
                                                    elif 'calm' in dominant_emotion[0].lower() or 'neutral' in dominant_emotion[0].lower():
                                                        st.info("🎯 Good composure! Consider adding more vocal variety and energy.")
                                                
                                                st.success("✅ Voice emotion analysis complete!")
                                            else:
                                                st.warning("⚠️ No emotions detected. Try speaking more clearly or with more expression.")
                                        
                                    except Exception as e:
                                        st.error(f"❌ Voice analysis failed: {e}")
                                        with st.expander("🔧 Technical Details"):
                                            import traceback
                                            st.code(traceback.format_exc())
                        
                        # Combined analysis button
                        if voice_answer and st.button("🚀 Complete Analysis", type="primary", use_container_width=True):
                            st.markdown("### 🔄 Processing Complete Analysis...")
                            
                            # Get current state
                            voice_transcription = st.session_state.get('voice_transcription', None)
                            voice_emotions = st.session_state.get('voice_emotions', None)
                            
                            # Process transcription if not already done
                            if not voice_transcription:
                                with st.spinner("Step 1: Transcribing voice..."):
                                    try:
                                        from src.speech_to_text import WhisperTranscriber
                                        transcriber = WhisperTranscriber()
                                        transcription_result = transcriber.transcribe_audio(voice_answer)
                                        
                                        if transcription_result and 'text' in transcription_result:
                                            st.session_state.voice_transcription = transcription_result['text']
                                            st.session_state.transcription_confidence = transcription_result.get('confidence', 0.0)
                                            st.success("✅ Transcription completed!")
                                        else:
                                            st.error("❌ Transcription failed")
                                            st.stop()
                                    except Exception as e:
                                        st.error(f"❌ Transcription error: {e}")
                                        st.stop()
                            
                            # Process emotions if not already done
                            if not voice_emotions:
                                with st.spinner("Step 2: Analyzing emotions..."):
                                    try:
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                            tmp_file.write(voice_answer)
                                            tmp_file.flush()
                                            tmp_path = tmp_file.name
                                        
                                        time.sleep(0.1)
                                        predictions = analyze_with_hume(tmp_path)
                                        
                                        try:
                                            os.unlink(tmp_path)
                                        except (PermissionError, FileNotFoundError):
                                            pass
                                        
                                        if predictions:
                                            emotions = extract_emotions(predictions)
                                            st.session_state.voice_emotions = emotions
                                            st.success("✅ Emotion analysis completed!")
                                        else:
                                            st.error("❌ Emotion analysis failed")
                                            st.stop()
                                    except Exception as e:
                                        st.error(f"❌ Emotion analysis error: {e}")
                                        st.stop()
                            
                            st.success("🎉 Complete analysis finished!")
                            st.rerun()
                
                # Process answer (either text or voice)
                user_answer = st.session_state.get('user_answer', '')
                voice_answer = st.session_state.get('voice_answer', None)
                voice_emotions = st.session_state.get('voice_emotions', None)
                voice_transcription = st.session_state.get('voice_transcription', None)
                transcription_confidence = st.session_state.get('transcription_confidence', 0.0)
                
                if user_answer or voice_answer:
                    # Clean answer analysis section
                    st.markdown("---")
                    st.markdown("### 🧠 Answer Analysis")
                    
                    # Analysis summary in clean cards
                    analysis_cols = st.columns(2)
                    
                    with analysis_cols[0]:
                        if user_answer:
                            st.markdown("#### 📝 Text Analysis")
                            st.info("**Answer Length:** " + ("✅ Good" if len(user_answer) > 100 else "⚠️ Could be more detailed"))
                            st.info("**Specificity:** " + ("✅ Good" if any(word in user_answer.lower() for word in ['because', 'example', 'when', 'result']) else "⚠️ Could use more examples"))
                            st.info("**STAR Method:** " + ("✅ Good" if any(word in user_answer.lower() for word in ['situation', 'task', 'action', 'result']) else "⚠️ Consider using STAR format"))
                        
                        if voice_answer:
                            st.markdown("#### 🎤 Voice Analysis")
                            st.success("**Recording:** ✅ Captured and analyzed")
                            if voice_transcription:
                                st.success("**Transcription:** ✅ Completed")
                                st.info(f"**Confidence:** {transcription_confidence:.2f}")
                    
                    with analysis_cols[1]:
                        if voice_emotions:
                            st.markdown("#### 🎵 Emotion Insights")
                            dominant = voice_emotions[0] if voice_emotions else None
                            if dominant:
                                emotion_name = dominant['name']
                                emotion_score = dominant['score']
                                
                                if emotion_score > 0.7:
                                    st.success(f"🔥 **High {emotion_name.title()}** - Very expressive!")
                                elif emotion_score > 0.4:
                                    st.info(f"🔸 **Medium {emotion_name.title()}** - Good balance")
                                else:
                                    st.warning(f"🔹 **Low {emotion_name.title()}** - Could be more expressive")
                        
                        # Analysis status
                        if voice_transcription and voice_emotions:
                            st.success("🎯 **Complete Analysis:** Voice + Text + Emotions")
                        elif voice_transcription or voice_emotions:
                            st.info("🔄 **Partial Analysis:** Some components ready")
                        else:
                            st.info("💡 **Next Steps:** Process voice for comprehensive analysis")
                    
                    # Intelligent Coaching Feedback
                    if voice_transcription and voice_emotions:
                        st.markdown("---")
                        st.markdown("### 🎯 AI Coaching")
                        
                        if st.button("🧠 Get AI Coaching Report", type="primary", use_container_width=True):
                            with st.spinner("🤖 AI is analyzing your answer and providing personalized coaching..."):
                                try:
                                    from src.coaching import CoachingAgent, CoachingContext
                                    
                                    current_question = st.session_state.get('current_question', {})
                                    question_text = current_question.get('question', 'Interview question')
                                    question_category = current_question.get('category', 'Behavioral')
                                    
                                    answer_duration = len(voice_transcription.split()) / 150.0
                                    
                                    context = CoachingContext(
                                        question_text=question_text,
                                        question_category=question_category,
                                        user_answer=voice_transcription,
                                        voice_emotions=voice_emotions,
                                        transcription_confidence=transcription_confidence,
                                        answer_duration=answer_duration
                                    )
                                    
                                    coaching_agent = CoachingAgent()
                                    feedback = coaching_agent.analyze_answer(context)
                                    
                                    st.session_state.coaching_feedback = feedback
                                    st.session_state.coaching_agent = coaching_agent
                                    
                                    # Check if AI was used successfully
                                    if hasattr(coaching_agent, 'llm_client') and coaching_agent.llm_client.test_connection():
                                        st.success("🎉 AI Coaching Report Generated with LLM!")
                                    else:
                                        st.success("🎉 Coaching Report Generated (Rule-based fallback)")
                                    
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Coaching analysis failed: {e}")
                                    with st.expander("🔧 Technical Details"):
                                        import traceback
                                        st.code(traceback.format_exc())
                    
                    # Display coaching feedback if available
                    if 'coaching_feedback' in st.session_state:
                        feedback = st.session_state.coaching_feedback
                        
                        st.markdown("---")
                        
                        # Professional Report Header
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
                            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 600;">📊 INTERVIEW PERFORMANCE ANALYSIS REPORT</h2>
                            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem;">AI-Powered Executive Coaching Assessment</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # AI Status indicator
                        if hasattr(st.session_state, 'coaching_agent') and st.session_state.coaching_agent.llm_client.test_connection():
                            st.markdown("""
                            <div style="background: #e8f5e8; border-left: 4px solid #28a745; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
                                <p style="margin: 0; color: #155724; font-weight: 500;">🤖 <strong>AI-Powered Analysis</strong> - Real-time intelligent coaching feedback generated by advanced language model</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
                                <p style="margin: 0; color: #856404; font-weight: 500;">📋 <strong>Rule-Based Analysis</strong> - Standard coaching feedback using predefined assessment criteria</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Executive Summary with Performance Metrics
                        st.markdown("""
                        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; border: 1px solid #dee2e6; margin-bottom: 1rem;">
                            <h3 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.4rem;">📈 PERFORMANCE METRICS</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Performance metrics in professional grid
                        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)  # Small spacer to move metrics up
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.markdown("""
                            <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; border: 1px solid #dee2e6;">
                                <div style="font-size: 2rem; font-weight: bold; color: #2c3e50;">{}</div>
                                <div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.5rem;">OVERALL SCORE</div>
                            </div>
                            """.format(feedback.overall_score), unsafe_allow_html=True)
                        with metric_cols[1]:
                            st.markdown("""
                            <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; border: 1px solid #dee2e6;">
                                <div style="font-size: 2rem; font-weight: bold; color: #2c3e50;">{}</div>
                                <div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.5rem;">CONTENT QUALITY</div>
                            </div>
                            """.format(feedback.content_score), unsafe_allow_html=True)
                        with metric_cols[2]:
                            st.markdown("""
                            <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; border: 1px solid #dee2e6;">
                                <div style="font-size: 2rem; font-weight: bold; color: #2c3e50;">{}</div>
                                <div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.5rem;">EMOTIONAL INTELLIGENCE</div>
                            </div>
                            """.format(feedback.emotion_score), unsafe_allow_html=True)
                        with metric_cols[3]:
                            st.markdown("""
                            <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; border: 1px solid #dee2e6;">
                                <div style="font-size: 2rem; font-weight: bold; color: #2c3e50;">{}</div>
                                <div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.5rem;">DELIVERY SKILLS</div>
                            </div>
                            """.format(feedback.delivery_score), unsafe_allow_html=True)
                        
                        # Professional feedback sections
                        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Spacer before feedback sections
                        feedback_cols = st.columns(2)
                        
                        with feedback_cols[0]:
                            st.markdown("""
                            <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #dee2e6; margin-bottom: 2rem;">
                                <h4 style="color: #28a745; margin-bottom: 1rem; font-size: 1.2rem;">✅ KEY STRENGTHS</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for strength in feedback.strengths:
                                st.markdown(f"""
                                <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.75rem; border-left: 3px solid #28a745;">
                                    <p style="margin: 0; color: #155724;">{strength}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #dee2e6; margin-bottom: 2rem;">
                                <h4 style="color: #dc3545; margin-bottom: 1rem; font-size: 1.2rem;">🎯 IMPROVEMENT AREAS</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for area in feedback.areas_for_improvement:
                                st.markdown(f"""
                                <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.75rem; border-left: 3px solid #dc3545;">
                                    <p style="margin: 0; color: #721c24;">{area}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with feedback_cols[1]:
                            st.markdown("""
                            <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #dee2e6; margin-bottom: 2rem;">
                                <h4 style="color: #17a2b8; margin-bottom: 1rem; font-size: 1.2rem;">💡 STRATEGIC RECOMMENDATIONS</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for suggestion in feedback.specific_suggestions:
                                st.markdown(f"""
                                <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.75rem; border-left: 3px solid #17a2b8;">
                                    <p style="margin: 0; color: #0c5460;">{suggestion}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            if feedback.star_method_feedback:
                                st.markdown("""
                                <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #dee2e6; margin-bottom: 2rem;">
                                    <h4 style="color: #6f42c1; margin-bottom: 1rem; font-size: 1.2rem;">⭐ STAR METHOD GUIDANCE</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 6px; border-left: 3px solid #6f42c1;">
                                    <p style="margin: 0; color: #4a148c;">{feedback.star_method_feedback}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Additional professional sections
                        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Spacer before additional sections
                        
                        if feedback.emotion_coaching:
                            st.markdown("""
                            <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #dee2e6; margin-bottom: 2rem;">
                                <h4 style="color: #fd7e14; margin-bottom: 1rem; font-size: 1.2rem;">🧠 EMOTIONAL INTELLIGENCE INSIGHTS</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 6px; border-left: 3px solid #fd7e14;">
                                <p style="margin: 0; color: #a0522d;">{feedback.emotion_coaching}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Development roadmap
                        st.markdown("""
                        <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #dee2e6; margin-bottom: 2rem;">
                            <h4 style="color: #20c997; margin-bottom: 1rem; font-size: 1.2rem;">🚀 DEVELOPMENT ROADMAP</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        roadmap_cols = st.columns(3)
                        with roadmap_cols[0]:
                            st.markdown("""
                            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #dee2e6;">
                                <h5 style="color: #20c997; margin-bottom: 0.5rem;">IMMEDIATE (24-48h)</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, step in enumerate(feedback.next_steps[:2]):
                                st.markdown(f"""
                                <div style="background: white; padding: 0.5rem; border-radius: 4px; margin-bottom: 0.25rem; border-left: 2px solid #20c997;">
                                    <p style="margin: 0; font-size: 0.9rem; color: #495057;">{step}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with roadmap_cols[1]:
                            st.markdown("""
                            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #dee2e6;">
                                <h5 style="color: #20c997; margin-bottom: 0.5rem;">SHORT-TERM (1 week)</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, step in enumerate(feedback.next_steps[2:4]):
                                st.markdown(f"""
                                <div style="background: white; padding: 0.5rem; border-radius: 4px; margin-bottom: 0.25rem; border-left: 2px solid #20c997;">
                                    <p style="margin: 0; font-size: 0.9rem; color: #495057;">{step}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with roadmap_cols[2]:
                            st.markdown("""
                            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #dee2e6;">
                                <h5 style="color: #20c997; margin-bottom: 0.5rem;">LONG-TERM (1 month)</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, step in enumerate(feedback.next_steps[4:6]):
                                st.markdown(f"""
                                <div style="background: white; padding: 0.5rem; border-radius: 4px; margin-bottom: 0.25rem; border-left: 2px solid #20c997;">
                                    <p style="margin: 0; font-size: 0.9rem; color: #495057;">{step}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Confidence boost section
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
                            <h4 style="color: white; margin-bottom: 1rem; font-size: 1.2rem;">💪 EXECUTIVE CONFIDENCE BOOST</h4>
                            <p style="margin: 0; font-size: 1.1rem; opacity: 0.95;">{}</p>
                        </div>
                        """.format(feedback.confidence_boost), unsafe_allow_html=True)
                        
                        # Professional action buttons
                        action_cols = st.columns([3, 1, 1])
                        with action_cols[1]:
                            if st.button("📊 Export Report", type="secondary", use_container_width=True):
                                st.info("Export functionality coming soon!")
                        
                        with action_cols[2]:
                            if st.button("🔄 New Session", type="primary", use_container_width=True):
                                if 'coaching_feedback' in st.session_state:
                                    del st.session_state.coaching_feedback
                                st.rerun()
                    
                    # Clean coaching preview
                    if voice_transcription or voice_emotions:
                        st.markdown("---")
                        st.markdown("### 🚀 Coaching Preview")
                        
                        coaching_features = []
                        if user_answer:
                            coaching_features.append("📝 Text Content Analysis")
                        if voice_transcription:
                            coaching_features.append("🎤 Voice Transcription Analysis")
                        if voice_emotions:
                            coaching_features.append("🎵 Emotion Pattern Recognition")
                        
                        if coaching_features:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea;">
                                <h4 style="margin-bottom: 1rem; color: #667eea;">🎯 Your Coaching Session Will Include:</h4>
                                <ul style="margin: 0; padding-left: 1.5rem;">
                            """, unsafe_allow_html=True)
                            
                            for feature in coaching_features:
                                st.markdown(f"<li>{feature}</li>", unsafe_allow_html=True)
                            
                            st.markdown("""
                                </ul>
                                <p style="margin-top: 1rem; font-style: italic; color: #666;">
                                    💡 <strong>Pro Tip:</strong> The more data you provide, the better your personalized coaching will be!
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
        
        else:
            # Clean welcome message
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%); border-radius: 15px; border: 2px dashed #667eea;">
                <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Ready to Practice?</h3>
                <p style="color: #666; font-size: 1.1rem; margin-bottom: 0.5rem;">Select your role and click "Get Question" to start practicing interview questions</p>
                <p style="color: #999; font-size: 0.9rem; margin: 0;">Each session provides unique questions with no repeats until all are completed</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Interview question system not available. Please check the logs for errors.")

# Footer
st.markdown("""
<div class="footer">
    <p>Powered by <strong>Hume AI</strong> • Built with ❤️ for emotional intelligence</p>
    <p style="font-size: 0.8rem; color: #999;">
        Enterprise-grade emotion analysis platform • Secure • Reliable • Accurate
    </p>
</div>
""", unsafe_allow_html=True)