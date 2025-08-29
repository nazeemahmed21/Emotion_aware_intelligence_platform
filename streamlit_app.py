"""
Streamlit frontend for Emotion-Aware Voice Feedback Bot
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
import tempfile
import os
import logging
import io

# Import our pipeline components
from src.pipeline import get_pipeline
from config import UI_CONFIG, EMOTION_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .emotion-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive { 
        background-color: #d4edda !important; 
        border-left-color: #28a745 !important;
    }
    .negative { 
        background-color: #f8d7da !important; 
        border-left-color: #dc3545 !important;
    }
    .neutral { 
        background-color: #e2e3e5 !important; 
        border-left-color: #6c757d !important;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .processing-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None

@st.cache_resource
def load_pipeline():
    """Load and cache the pipeline"""
    try:
        with st.spinner("🔄 Loading AI models... This may take a few minutes on first run."):
            pipeline = get_pipeline()
            return pipeline
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None

def audio_input_component():
    """Component for audio input (recording or file upload)"""
    st.subheader("🎤 Audio Input")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload File", "🎙️ Record Audio"])
    
    audio_data = None
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
            help="Upload an audio file to analyze emotions and generate responses"
        )
        
        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Display audio player
            st.audio(audio_data, format='audio/wav')
    
    with tab2:
        st.info("🎙️ Audio recording functionality requires additional setup. For now, please use file upload.")
        st.markdown("""
        **To enable audio recording:**
        1. Install: `pip install streamlit-webrtc audio-recorder-streamlit`
        2. Configure microphone permissions in your browser
        3. Restart the application
        """)
    
    return audio_data

def display_audio_waveform(audio_data, sample_rate=16000):
    """Display audio waveform visualization"""
    try:
        # Convert audio bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Create time axis
        time_axis = np.arange(len(audio_float)) / sample_rate
        
        # Create waveform plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=audio_float,
            mode='lines',
            name='Waveform',
            line=dict(color='#1f77b4', width=1)
        ))
        
        fig.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error visualizing audio: {e}")

def display_emotion_results(result):
    """Display emotion detection results with visualizations"""
    if not result.get('success', False):
        st.error("❌ Processing failed")
        if 'error' in result:
            st.error(f"Error: {result['error']}")
        return
    
    emotion = result.get('emotion', 'neutral')
    confidence = result.get('emotion_confidence', 0.0)
    
    # Emotion color and emoji mapping
    emotion_config = {
        'happy': {'color': '#28a745', 'emoji': '😊', 'class': 'positive'},
        'excited': {'color': '#ffc107', 'emoji': '🤩', 'class': 'positive'},
        'calm': {'color': '#17a2b8', 'emoji': '😌', 'class': 'positive'},
        'sad': {'color': '#6f42c1', 'emoji': '😢', 'class': 'negative'},
        'angry': {'color': '#dc3545', 'emoji': '😠', 'class': 'negative'},
        'fearful': {'color': '#fd7e14', 'emoji': '😰', 'class': 'negative'},
        'neutral': {'color': '#6c757d', 'emoji': '😐', 'class': 'neutral'}
    }
    
    config = emotion_config.get(emotion, emotion_config['neutral'])
    
    # Main emotion display
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h2 style="text-align: center; margin: 0;">{config['emoji']}</h2>
            <p style="text-align: center; margin: 0; font-size: 1.2em; font-weight: bold;">
                {emotion.capitalize()}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence"},
            delta = {'reference': 70},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': config['color']},
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
        
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        processing_time = result.get('processing_time', 0.0)
        audio_duration = result.get('audio_duration', 0.0)
        
        st.markdown(f"""
        <div class="metric-container">
            <p><strong>Processing:</strong> {processing_time:.2f}s</p>
            <p><strong>Audio:</strong> {audio_duration:.2f}s</p>
            <p><strong>Ratio:</strong> {processing_time/max(audio_duration, 0.1):.1f}x</p>
        </div>
        """, unsafe_allow_html=True)
    
    # All emotions breakdown
    if 'steps' in result and 'emotion_recognition' in result['steps']:
        all_emotions = result['steps']['emotion_recognition'].get('all_emotions', [])
        
        if len(all_emotions) > 1:
            with st.expander("📊 All Emotion Scores"):
                emotions_df = pd.DataFrame(all_emotions)
                
                fig = px.bar(
                    emotions_df, 
                    x='emotion', 
                    y='confidence',
                    title="Emotion Detection Scores",
                    color='confidence',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def display_transcription_and_response(result):
    """Display transcription and AI response"""
    if not result.get('success', False):
        return
    
    transcription = result.get('transcription', '')
    response = result.get('response', '')
    emotion = result.get('emotion', 'neutral')
    
    # Determine card class based on emotion
    emotion_class = 'neutral'
    if emotion in ['happy', 'excited', 'calm']:
        emotion_class = 'positive'
    elif emotion in ['sad', 'angry', 'fearful']:
        emotion_class = 'negative'
    
    # Display transcription
    st.subheader("📝 What You Said")
    st.markdown(f"""
    <div class="emotion-card {emotion_class}">
        <h4>🗣️ Transcription:</h4>
        <p style="font-size: 1.1em; font-style: italic;">"{transcription}"</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display AI response
    st.subheader("🤖 AI Response")
    st.markdown(f"""
    <div class="emotion-card {emotion_class}">
        <h4>💬 Empathetic Response:</h4>
        <p style="font-size: 1.1em;">{response}</p>
    </div>
    """, unsafe_allow_html=True)

def display_conversation_history():
    """Display conversation history"""
    if not st.session_state.conversation_history:
        st.info("💭 No conversations yet. Upload an audio file to get started!")
        return
    
    st.subheader("💬 Conversation History")
    
    for i, conversation in enumerate(reversed(st.session_state.conversation_history)):
        with st.expander(f"Conversation {len(st.session_state.conversation_history) - i} - {conversation.get('emotion', 'unknown').capitalize()}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🗣️ User:**")
                st.write(conversation.get('transcription', 'N/A'))
                st.markdown(f"**🎭 Emotion:** {conversation.get('emotion', 'unknown').capitalize()}")
                st.markdown(f"**📊 Confidence:** {conversation.get('confidence', 0.0):.1%}")
            
            with col2:
                st.markdown("**🤖 AI Response:**")
                st.write(conversation.get('response', 'N/A'))
                st.markdown(f"**⏰ Time:** {conversation.get('timestamp', 'unknown')}")

def sidebar_configuration():
    """Sidebar with configuration options"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Pipeline status
        st.subheader("🔧 System Status")
        if st.session_state.pipeline:
            status = st.session_state.pipeline.get_pipeline_status()
            
            for component, is_ready in status.items():
                if component != 'ready':
                    icon = "✅" if is_ready else "❌"
                    st.write(f"{icon} {component.replace('_', ' ').title()}")
        
        st.divider()
        
        # Model information
        st.subheader("🤖 Model Information")
        st.write("**Speech-to-Text:** Whisper (base)")
        st.write("**Emotion Recognition:** Wav2Vec2")
        st.write("**Response Generation:** Ollama LLM")
        
        st.divider()
        
        # Conversation management
        st.subheader("💬 Conversation")
        conversation_count = len(st.session_state.conversation_history)
        st.write(f"**Messages:** {conversation_count}")
        
        if conversation_count > 0:
            if st.button("🗑️ Clear History"):
                st.session_state.conversation_history = []
                if st.session_state.pipeline:
                    st.session_state.pipeline.clear_conversation_history()
                st.success("History cleared!")
                st.rerun()
        
        st.divider()
        
        # Export functionality
        if conversation_count > 0:
            st.subheader("📊 Export Data")
            if st.button("📥 Download CSV"):
                df = pd.DataFrame(st.session_state.conversation_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 Download",
                    data=csv,
                    file_name=f"emotion_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Load pipeline
    if st.session_state.pipeline is None:
        st.session_state.pipeline = load_pipeline()
    
    # Main header
    st.markdown('<h1 class="main-header">🎭 Emotion-Aware Voice Bot</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    sidebar_configuration()
    
    # Check if pipeline is ready
    if st.session_state.pipeline is None:
        st.error("❌ Failed to load AI models. Please check the setup instructions.")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Audio input section
        audio_data = audio_input_component()
        
        if audio_data:
            # Display audio waveform
            st.subheader("📊 Audio Visualization")
            display_audio_waveform(audio_data)
            
            # Process audio button
            if st.button("🎯 Analyze Emotion & Generate Response", type="primary", use_container_width=True):
                st.session_state.processing = True
                
                # Process audio through pipeline
                with st.spinner("🔄 Processing audio... This may take a moment."):
                    result = st.session_state.pipeline.process_audio(audio_data, include_features=True)
                
                if result.get('success', False):
                    # Add to conversation history
                    conversation_entry = {
                        'transcription': result.get('transcription', ''),
                        'emotion': result.get('emotion', 'neutral'),
                        'confidence': result.get('emotion_confidence', 0.0),
                        'response': result.get('response', ''),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'processing_time': result.get('processing_time', 0.0)
                    }
                    st.session_state.conversation_history.append(conversation_entry)
                    
                    # Display results
                    st.success("✅ Processing completed!")
                    
                    # Display emotion results
                    display_emotion_results(result)
                    
                    # Display transcription and response
                    display_transcription_and_response(result)
                    
                else:
                    st.error("❌ Processing failed. Please try again.")
                    if 'error' in result:
                        st.error(f"Error details: {result['error']}")
                
                st.session_state.processing = False
    
    with col2:
        # Processing steps info
        st.subheader("🔄 Processing Steps")
        st.markdown("""
        1. **🎵 Audio Loading** - Load and preprocess audio
        2. **📝 Speech-to-Text** - Transcribe using Whisper
        3. **🎭 Emotion Detection** - Analyze emotional content
        4. **🤖 Response Generation** - Create empathetic response
        """)
        
        # Tips
        st.subheader("💡 Tips")
        st.markdown("""
        - **Clear speech** works best for transcription
        - **2-10 seconds** of audio is optimal
        - **Express emotions** clearly for better detection
        - **Multiple languages** supported by Whisper
        """)
    
    # Conversation history
    st.divider()
    display_conversation_history()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🎭 Built with Streamlit • 🤖 Powered by AI • 💝 Made with empathy</p>
        <p><small>Emotion-Aware Voice Feedback Bot v1.0</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()