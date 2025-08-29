5. Streamlit Frontend Integration
Overview
Streamlit provides a simple and powerful way to create interactive web applications for your emotion-aware voice feedback bot. This section covers the complete frontend integration with all previous components.

Installation
bash
pip install streamlit streamlit-webrtc audio-recorder-streamlit
Basic Streamlit App Structure
Main Application Layout
python
# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
import tempfile
import os

# Import your backend components
from whisper_transcription import transcribe_audio
from audio_processing import extract_audio_features
from emotion_detection import detect_emotion
from ollama_response import generate_empathetic_response

# Page configuration
st.set_page_config(
    page_title="Emotion-Aware Voice Bot",
    page_icon="🎭",
    layout="wide",
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
    }
    .emotion-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive { background-color: #d4edda !important; }
    .negative { background-color: #f8d7da !important; }
    .neutral { background-color: #e2e3e5 !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
Audio Recording Components
Audio Recorder Integration
python
def audio_recorder_component():
    """Component for recording audio"""
    try:
        from audio_recorder_streamlit import audio_recorder
        
        st.subheader("🎤 Record Your Voice")
        
        # Audio recorder
        audio_bytes = audio_recorder(
            text="Click to start recording",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
        )
        
        # File uploader as alternative
        uploaded_file = st.file_uploader(
            "Or upload an audio file", 
            type=['wav', 'mp3', 'm4a', 'ogg']
        )
        
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
        
        return audio_bytes
        
    except ImportError:
        st.warning("Audio recorder component not available. Using file uploader only.")
        uploaded_file = st.file_uploader(
            "Upload an audio file", 
            type=['wav', 'mp3', 'm4a', 'ogg']
        )
        return uploaded_file.read() if uploaded_file else None
Audio Visualization
python
def plot_audio_waveform(audio_bytes, sample_rate=16000):
    """Plot audio waveform"""
    try:
        import librosa
        import io
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0  # Normalize
        
        # Create waveform plot
        fig = px.line(
            x=np.arange(len(audio_float)) / sample_rate,
            y=audio_float,
            labels={'x': 'Time (seconds)', 'y': 'Amplitude'},
            title="Audio Waveform"
        )
        fig.update_layout(height=200, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error visualizing audio: {e}")
Emotion Analysis Display
Emotion Visualization
python
def display_emotion_results(emotion, confidence, features):
    """Display emotion detection results"""
    
    # Emotion color mapping
    emotion_colors = {
        'happy': '🟢', 'excited': '🟢', 'joy': '🟢',
        'sad': '🔵', 'angry': '🔴', 'fear': '🟣',
        'neutral': '⚪', 'calm': '🟢'
    }
    
    # Display emotion with confidence
    col1, col2 = st.columns([1, 2])
    
    with col1:
        emoji = emotion_colors.get(emotion, '⚪')
        st.metric(
            label="Detected Emotion",
            value=f"{emoji} {emotion.capitalize()}",
            delta=f"{confidence:.1%} confidence"
        )
    
    with col2:
        # Progress bar for confidence
        st.progress(float(confidence))
        st.caption(f"Confidence level: {confidence:.1%}")
    
    # Feature visualization
    if features:
        with st.expander("📊 Audio Features Analysis"):
            feature_df = pd.DataFrame.from_dict(features, orient='index', columns=['Value'])
            st.dataframe(feature_df.style.highlight_max(axis=0), use_container_width=True)
Conversation History Display
python
def display_conversation_history():
    """Display conversation history with emotion context"""
    if not st.session_state.conversation_history:
        st.info("No conversations yet. Record some audio to get started!")
        return
    
    st.subheader("💬 Conversation History")
    
    for i, conversation in enumerate(st.session_state.conversation_history):
        # Determine emotion class for styling
        emotion_class = "neutral"
        if conversation['emotion'] in ['happy', 'excited', 'calm']:
            emotion_class = "positive"
        elif conversation['emotion'] in ['sad', 'angry', 'fear']:
            emotion_class = "negative"
        
        # Display conversation card
        with st.container():
            st.markdown(f"""
            <div class="emotion-card {emotion_class}">
                <strong>🗣️ User ({conversation['emotion'].capitalize()}):</strong> {conversation['user_input']}<br>
                <strong>🤖 AI Response:</strong> {conversation['ai_response']}<br>
                <small>⏰ {conversation['timestamp']} | Confidence: {conversation['confidence']:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Add delete button for each conversation
            if st.button("🗑️", key=f"delete_{i}"):
                st.session_state.conversation_history.pop(i)
                st.rerun()
Real-time Processing Functions
Audio Processing Pipeline
python
def process_audio_pipeline(audio_bytes):
    """Complete audio processing pipeline"""
    
    # Show processing status
    with st.status("🔍 Analyzing audio...", expanded=True) as status:
        st.write("Transcribing speech...")
        
        # Step 1: Transcribe audio
        try:
            transcription = transcribe_audio(audio_bytes)
            st.write(f"📝 Transcription: {transcription}")
        except Exception as e:
            st.error(f"Transcription error: {e}")
            return None
        
        # Step 2: Extract features
        st.write("Extracting audio features...")
        try:
            features = extract_audio_features(audio_bytes)
            st.write("✅ Features extracted")
        except Exception as e:
            st.error(f"Feature extraction error: {e}")
            return None
        
        # Step 3: Detect emotion
        st.write("Detecting emotion...")
        try:
            emotion, confidence = detect_emotion(features)
            st.write(f"🎭 Detected emotion: {emotion} ({confidence:.1%})")
        except Exception as e:
            st.error(f"Emotion detection error: {e}")
            return None
        
        # Step 4: Generate response
        st.write("Generating response...")
        try:
            response = generate_empathetic_response(transcription, emotion, confidence)
            st.write("💬 Response generated")
        except Exception as e:
            st.error(f"Response generation error: {e}")
            return None
        
        status.update(label="Analysis complete!", state="complete")
    
    return {
        'transcription': transcription,
        'emotion': emotion,
        'confidence': confidence,
        'response': response,
        'features': features,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
Main Application Layout
Sidebar Configuration
python
def sidebar_config():
    """Application sidebar with configuration options"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        model_option = st.selectbox(
            "LLM Model",
            ["llama2", "mistral", "mixtral"],
            help="Choose which language model to use for responses"
        )
        
        # Emotion detection sensitivity
        sensitivity = st.slider(
            "Emotion Detection Sensitivity",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            help="Higher values make emotion detection more sensitive but less specific"
        )
        
        # Response style
        response_style = st.selectbox(
            "Response Style",
            ["Empathetic", "Professional", "Friendly", "Concise"],
            help="Choose the style of AI responses"
        )
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation History"):
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")
        
        st.divider()
        
        # System info
        st.subheader("System Information")
        st.write(f"**Model:** {model_option}")
        st.write(f"**Sensitivity:** {sensitivity:.1f}")
        st.write(f"**Style:** {response_style}")
        st.write(f"**Conversations:** {len(st.session_state.conversation_history)}")
        
        return model_option, sensitivity, response_style
Main Content Area
python
def main_content():
    """Main content area of the application"""
    st.markdown('<h1 class="main-header">🎭 Emotion-Aware Voice Bot</h1>', unsafe_allow_html=True)
    
    # Record or upload audio
    audio_bytes = audio_recorder_component()
    
    if audio_bytes:
        # Display audio waveform
        plot_audio_waveform(audio_bytes)
        
        # Process audio when button is clicked
        if st.button("🎯 Analyze Emotion & Generate Response", type="primary"):
            st.session_state.processing = True
            
            # Process audio through pipeline
            result = process_audio_pipeline(audio_bytes)
            
            if result:
                # Add to conversation history
                st.session_state.conversation_history.append({
                    'user_input': result['transcription'],
                    'emotion': result['emotion'],
                    'confidence': result['confidence'],
                    'ai_response': result['response'],
                    'timestamp': result['timestamp']
                })
                
                # Display results
                st.divider()
                display_emotion_results(
                    result['emotion'], 
                    result['confidence'], 
                    result['features']
                )
                
                # Display AI response
                st.subheader("💬 AI Response")
                st.success(result['response'])
                
                # Text-to-speech option
                if st.button("🔊 Speak Response"):
                    # This would integrate with your TTS system
                    st.info("Text-to-speech functionality would play here")
            
            st.session_state.processing = False
    
    # Display conversation history
    st.divider()
    display_conversation_history()
    
    # Export functionality
    if st.session_state.conversation_history:
        st.divider()
        if st.button("📊 Export Conversation Data"):
            export_conversation_data()
Export Functionality
python
def export_conversation_data():
    """Export conversation data to CSV"""
    df = pd.DataFrame(st.session_state.conversation_history)
    
    # Convert DataFrame to CSV
    csv = df.to_csv(index=False)
    
    # Create download button
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"emotion_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Show data preview
    with st.expander("📋 Data Preview"):
        st.dataframe(df, use_container_width=True)
Complete Application Integration
Main Function
python
def main():
    """Main application function"""
    
    # Load configuration from sidebar
    model_config, sensitivity, response_style = sidebar_config()
    
    # Set global configuration
    st.session_state.model_config = model_config
    st.session_state.sensitivity = sensitivity
    st.session_state.response_style = response_style
    
    # Main content
    main_content()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with 🎯 Streamlit, 🎭 Emotion AI, and 💬 LLM Technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
Real-time Updates and Session Management
Auto-refresh and State Management
python
# Add this to your main app for real-time features
def setup_real_time_features():
    """Setup real-time features and auto-refresh"""
    
    # Auto-refresh every 30 seconds if processing
    if st.session_state.get('processing', False):
        time.sleep(0.1)
        st.rerun()
    
    # Keyboard shortcuts
    st.sidebar.markdown("---")
    st.sidebar.subheader("⌨️ Keyboard Shortcuts")
    st.sidebar.write("• **R**: Record audio")
    st.sidebar.write("• **A**: Analyze emotion")
    st.sidebar.write("• **C**: Clear history")
    
    # Add JavaScript for keyboard shortcuts
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.key === 'r' && !e.ctrlKey) {
            e.preventDefault();
            // Trigger record button
        }
        if (e.key === 'a' && !e.ctrlKey) {
            e.preventDefault();
            // Trigger analyze button
        }
    });
    </script>
    """, unsafe_allow_html=True)
Deployment Configuration
requirements.txt
txt
streamlit==1.28.0
streamlit-webrtc==0.47.0
audio-recorder-streamlit==0.0.6
openai-whisper==20231117
librosa==0.10.1
pyAudioAnalysis==0.3.14
ollama==0.1.6
numpy==1.24.3
pandas==2.0.3
plotly==5.15.0
scikit-learn==1.3.0
streamlit_config.toml
toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
serverAddress = "localhost"
serverPort = 8501

[theme]
base = "light"
Running the Application
Local Development
bash
streamlit run app.py
Production Deployment
bash
# Using Docker
docker build -t emotion-bot .
docker run -p 8501:8501 emotion-bot

# Using Hugging Face Spaces
# Upload your code to a GitHub repo and connect to Hugging Face Spaces
This comprehensive Streamlit frontend provides a complete user interface for your emotion-aware voice feedback bot, integrating all the backend components into a seamless, interactive experience.

