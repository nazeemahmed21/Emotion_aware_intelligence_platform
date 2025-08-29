#!/usr/bin/env python3
"""
Main Streamlit app for Emotion-Aware Voice Pipeline
"""
import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os
import sys
import traceback

# Add src to path
sys.path.append('src')
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="🎭 Emotion-Aware Voice Bot",
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

@st.cache_resource
def load_pipeline():
    """Load and cache the pipeline"""
    try:
        with st.spinner("🔄 Loading AI models... This may take a few minutes on first run."):
            from pipeline import EmotionAwareVoicePipeline
            pipeline = EmotionAwareVoicePipeline()
            return pipeline
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        st.error("Make sure all dependencies are installed and Ollama is running")
        return None

def display_emotion_results(result):
    """Display emotion detection results"""
    if not result.get('success', False):
        st.error("❌ Processing failed")
        if 'error' in result:
            st.error(f"Error: {result['error']}")
        return
    
    emotion = result.get('emotion', 'neutral')
    confidence = result.get('emotion_confidence', 0.0)
    
    # Emotion emoji mapping
    emotion_emojis = {
        'happy': '😊',
        'excited': '🤩',
        'calm': '😌',
        'sad': '😢',
        'angry': '😠',
        'fearful': '😰',
        'neutral': '😐'
    }
    
    emoji = emotion_emojis.get(emotion, '😐')
    
    # Display basic emotion results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎭 Detected Emotion", f"{emoji} {emotion.capitalize()}")
    
    with col2:
        st.metric("📊 Confidence", f"{confidence:.1%}")
    
    with col3:
        processing_time = result.get('processing_time', 0.0)
        st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
    
    # Display preprocessing info and comprehensive emotion analysis if available
    emotion_steps = result.get('steps', {}).get('emotion_recognition', {})
    if emotion_steps.get('model_type') == 'hume_ai':
        display_preprocessing_info(emotion_steps)
    
    # Display comprehensive emotion analysis if available
    if 'aggregated_analysis' in emotion_steps:
        display_aggregated_emotion_analysis(emotion_steps['aggregated_analysis'])

def display_preprocessing_info(emotion_steps):
    """Display audio preprocessing information"""
    if emotion_steps.get('model_type') == 'hume_ai':
        st.subheader("🔧 Audio Preprocessing Applied")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Format Conversion:**
            - ✅ WAV with Linear PCM
            - ✅ 16-bit depth
            - ✅ Mono channel
            """)
        
        with col2:
            st.markdown("""
            **Audio Optimization:**
            - ✅ 16kHz sample rate
            - ✅ Amplitude normalized
            - ✅ Duration optimized
            """)
        
        with col3:
            st.markdown("""
            **Hume AI Ready:**
            - ✅ < 10MB file size
            - ✅ Optimal duration
            - ✅ High quality encoding
            """)
        
        st.success("🎯 Your audio has been automatically optimized for the most accurate emotion analysis!")

def display_aggregated_emotion_analysis(analysis):
    """Display comprehensive aggregated emotion analysis"""
    if not analysis:
        st.warning("⚠️ No aggregated emotion analysis available")
        return
    
    st.subheader("🧠 Comprehensive Emotion Analysis")
    
    # Summary metrics
    summary = analysis.get('summary', {})
    total_emotions = summary.get('total_emotions_detected', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Total Emotions", total_emotions)
    
    with col2:
        dominant = summary.get('dominant_emotion', {})
        st.metric("👑 Dominant Emotion", f"{dominant.get('name', 'N/A')} ({dominant.get('score', 0):.2f})")
    
    with col3:
        st.metric("⚡ Overall Intensity", f"{summary.get('overall_intensity', 0):.2f}")
    
    with col4:
        sentiment_balance = summary.get('sentiment_balance', 0)
        sentiment_emoji = "😊" if sentiment_balance > 0.1 else "😢" if sentiment_balance < -0.1 else "😐"
        st.metric("⚖️ Sentiment Balance", f"{sentiment_emoji} {sentiment_balance:.2f}")
    
    # Show message if no emotions detected
    if total_emotions == 0:
        st.info("ℹ️ **No emotions detected in this audio.** This may be because:")
        st.write("• The audio doesn't contain clear speech")
        st.write("• The audio is too quiet or unclear")
        st.write("• The content is emotionally neutral")
        st.write("• Try uploading audio with clear emotional speech for better results")
        return
    
    # Top emotions
    top_emotions = analysis.get('top_emotions', [])
    if top_emotions:
        st.subheader("🏆 Top 10 Emotions Detected")
        
        # Create DataFrame for better display
        emotion_data = []
        for i, emotion in enumerate(top_emotions[:10]):
            intensity_emoji = "🔥" if emotion.get('intensity_level') == 'high' else "🔸" if emotion.get('intensity_level') == 'medium' else "🔹"
            emotion_data.append({
                'Rank': i + 1,
                'Emotion': f"{intensity_emoji} {emotion.get('name', 'Unknown')}",
                'Score': f"{emotion.get('mean_score', 0):.3f}",
                'Peak': f"{emotion.get('max_score', 0):.3f}",
                'Frequency': emotion.get('frequency', 0),
                'Intensity': emotion.get('intensity_level', 'unknown').title()
            })
        
        df = pd.DataFrame(emotion_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Intensity distribution
    intensity_dist = analysis.get('intensity_distribution', {})
    if intensity_dist:
        st.subheader("📊 Emotion Intensity Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_intensity = intensity_dist.get('high_intensity', {})
            st.markdown("### 🔥 High Intensity")
            st.metric("Count", high_intensity.get('count', 0))
            st.metric("Avg Score", f"{high_intensity.get('average_score', 0):.3f}")
            if high_intensity.get('emotions'):
                st.write("**Emotions:**")
                for emotion in high_intensity['emotions'][:5]:  # Show top 5
                    st.write(f"• {emotion}")
        
        with col2:
            medium_intensity = intensity_dist.get('medium_intensity', {})
            st.markdown("### 🔸 Medium Intensity")
            st.metric("Count", medium_intensity.get('count', 0))
            st.metric("Avg Score", f"{medium_intensity.get('average_score', 0):.3f}")
            if medium_intensity.get('emotions'):
                st.write("**Emotions:**")
                for emotion in medium_intensity['emotions'][:5]:
                    st.write(f"• {emotion}")
        
        with col3:
            low_intensity = intensity_dist.get('low_intensity', {})
            st.markdown("### 🔹 Low Intensity")
            st.metric("Count", low_intensity.get('count', 0))
            st.metric("Avg Score", f"{low_intensity.get('average_score', 0):.3f}")
            if low_intensity.get('emotions'):
                st.write("**Emotions:**")
                for emotion in low_intensity['emotions'][:5]:
                    st.write(f"• {emotion}")
    
    # Emotional categories
    categories = analysis.get('emotional_categories', {})
    if categories:
        st.subheader("🎨 Emotional Categories")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            positive = categories.get('positive', {})
            st.markdown("### 😊 Positive Emotions")
            st.metric("Total Score", f"{positive.get('total_score', 0):.3f}")
            pos_emotions = positive.get('emotions', {})
            if pos_emotions:
                st.write("**Top Positive:**")
                sorted_pos = sorted(pos_emotions.items(), key=lambda x: x[1].get('mean', 0), reverse=True)
                for emotion, stats in sorted_pos[:3]:
                    st.write(f"• {emotion}: {stats.get('mean', 0):.3f}")
        
        with col2:
            negative = categories.get('negative', {})
            st.markdown("### 😢 Negative Emotions")
            st.metric("Total Score", f"{negative.get('total_score', 0):.3f}")
            neg_emotions = negative.get('emotions', {})
            if neg_emotions:
                st.write("**Top Negative:**")
                sorted_neg = sorted(neg_emotions.items(), key=lambda x: x[1].get('mean', 0), reverse=True)
                for emotion, stats in sorted_neg[:3]:
                    st.write(f"• {emotion}: {stats.get('mean', 0):.3f}")
        
        with col3:
            neutral = categories.get('neutral', {})
            st.markdown("### 😐 Neutral")
            st.metric("Score", f"{neutral.get('score', 0):.3f}")
    
    # Hume AI Model Analysis
    model_analysis = analysis.get('model_analysis', {})
    if model_analysis:
        st.subheader("🧠 Hume AI Model Breakdown")
        st.markdown("*Hume AI uses three specialized models to analyze different aspects of emotional expression:*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prosody = model_analysis.get('prosody', {})
            st.markdown("### 🎵 Prosody Model")
            st.write(f"**Focus:** {prosody.get('description', 'N/A')}")
            st.metric("Emotions Detected", prosody.get('emotion_count', 0))
            st.metric("Avg Intensity", f"{prosody.get('avg_intensity', 0):.3f}")
            st.metric("Contribution", f"{prosody.get('contribution_percentage', 0):.1f}%")
            
            if prosody.get('top_emotions'):
                st.write("**Top Emotions:**")
                for emotion, score in prosody['top_emotions'][:3]:
                    st.write(f"• {emotion}: {score:.3f}")
        
        with col2:
            burst = model_analysis.get('burst', {})
            st.markdown("### ⚡ Burst Model")
            st.write(f"**Focus:** {burst.get('description', 'N/A')}")
            st.metric("Emotions Detected", burst.get('emotion_count', 0))
            st.metric("Avg Intensity", f"{burst.get('avg_intensity', 0):.3f}")
            st.metric("Contribution", f"{burst.get('contribution_percentage', 0):.1f}%")
            
            if burst.get('top_emotions'):
                st.write("**Top Emotions:**")
                for emotion, score in burst['top_emotions'][:3]:
                    st.write(f"• {emotion}: {score:.3f}")
        
        with col3:
            language = model_analysis.get('language', {})
            st.markdown("### 📝 Language Model")
            st.write(f"**Focus:** {language.get('description', 'N/A')}")
            st.metric("Emotions Detected", language.get('emotion_count', 0))
            st.metric("Avg Intensity", f"{language.get('avg_intensity', 0):.3f}")
            st.metric("Contribution", f"{language.get('contribution_percentage', 0):.1f}%")
            
            if language.get('top_emotions'):
                st.write("**Top Emotions:**")
                for emotion, score in language['top_emotions'][:3]:
                    st.write(f"• {emotion}: {score:.3f}")
        
        # Model summary
        model_summary = analysis.get('model_summary', {})
        if model_summary:
            st.info(f"🎯 **Primary Model**: {model_summary.get('primary_model', 'unknown').title()} | **Models Used**: {model_summary.get('total_models_used', 0)}/3")
    
    # Recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        st.subheader("💡 AI Insights & Recommendations")
        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {rec}")
    
    # Detailed analysis in expandable section
    with st.expander("🔍 Detailed Emotion Breakdown"):
        detailed_emotions = analysis.get('detailed_emotions', {})
        if detailed_emotions:
            # Create comprehensive emotion table
            detailed_data = []
            for emotion, stats in detailed_emotions.items():
                detailed_data.append({
                    'Emotion': emotion,
                    'Mean Score': f"{stats.get('mean', 0):.4f}",
                    'Max Score': f"{stats.get('max', 0):.4f}",
                    'Min Score': f"{stats.get('min', 0):.4f}",
                    'Std Dev': f"{stats.get('std', 0):.4f}",
                    'Count': stats.get('count', 0),
                    'Total': f"{stats.get('total', 0):.4f}"
                })
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df = detailed_df.sort_values('Mean Score', ascending=False)
            st.dataframe(detailed_df, use_container_width=True, hide_index=True)
        
        # Raw data info
        raw_data = analysis.get('raw_data', {})
        if raw_data:
            st.write("**Raw Analysis Data:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", raw_data.get('total_predictions', 0))
            with col2:
                st.metric("Unique Emotions", raw_data.get('unique_emotions', 0))
            with col3:
                score_range = raw_data.get('score_range', {})
                st.write(f"**Score Range:** {score_range.get('min', 0):.3f} - {score_range.get('max', 0):.3f}")

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

def sidebar_info():
    """Sidebar with information and controls"""
    with st.sidebar:
        st.title("⚙️ Controls")
        
        # Pipeline status
        st.subheader("🔧 System Status")
        if st.session_state.pipeline:
            st.success("✅ Pipeline loaded")
        else:
            st.error("❌ Pipeline not loaded")
        
        st.divider()
        
        # Model information
        st.subheader("🤖 AI Models")
        st.write("**Speech-to-Text:** Whisper")
        st.write("**Emotion Recognition:** Wav2Vec2")
        st.write("**Response Generation:** Ollama")
        
        st.divider()
        
        # Conversation management
        st.subheader("💬 Conversation")
        conversation_count = len(st.session_state.conversation_history)
        st.write(f"**Messages:** {conversation_count}")
        
        if conversation_count > 0:
            if st.button("🗑️ Clear History"):
                st.session_state.conversation_history = []
                st.success("History cleared!")
                st.rerun()
        
        st.divider()
        
        # Instructions
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Upload** an audio file (WAV, MP3, M4A)
        2. **Click** "Analyze Audio" to process
        3. **View** emotion detection and AI response
        4. **Check** conversation history below
        """)
        
        # Tips
        st.subheader("💡 Tips")
        st.markdown("""
        - Clear speech works best
        - 2-10 seconds optimal length
        - Express emotions clearly
        - Multiple languages supported
        """)

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🎭 Emotion-Aware Voice Bot</h1>', unsafe_allow_html=True)
    st.markdown("*Upload audio to detect emotions and get empathetic AI responses*")
    
    # Sidebar
    sidebar_info()
    
    # Load pipeline
    if st.session_state.pipeline is None:
        st.session_state.pipeline = load_pipeline()
    
    # Check if pipeline is ready
    if st.session_state.pipeline is None:
        st.error("❌ Failed to load AI models. Please check the setup:")
        st.markdown("""
        **Setup Requirements:**
        1. Install dependencies: `pip install -r requirements.txt`
        2. Start Ollama: `ollama serve`
        3. Install models: `ollama pull llama2`
        4. Restart this app
        """)
        st.stop()
    
    # Main content
    st.subheader("📁 Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
        help="Upload an audio file to analyze emotions and generate responses"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ Uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # Audio player
        st.audio(uploaded_file.read(), format='audio/wav')
        uploaded_file.seek(0)  # Reset file pointer
        
        # Audio validation (if using Hume AI)
        if hasattr(st.session_state.pipeline.emotion_recognizer, 'validate_audio_for_hume'):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Validate Audio Format", help="Check if audio meets Hume AI specifications"):
                    with st.spinner("Validating audio format..."):
                        # Save uploaded file temporarily for validation
                        import tempfile
                        import time
                        
                        tmp_file_path = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_file.flush()
                                tmp_file_path = tmp_file.name
                                uploaded_file.seek(0)  # Reset file pointer
                            
                            # Small delay to ensure file is released on Windows
                            time.sleep(0.1)
                            
                            # Validate the audio
                            validation = st.session_state.pipeline.emotion_recognizer.validate_audio_for_hume(tmp_file_path)
                            
                        finally:
                            # Clean up temp file with retry logic for Windows
                            if tmp_file_path:
                                try:
                                    import os
                                    time.sleep(0.1)  # Additional delay for Windows
                                    os.unlink(tmp_file_path)
                                except (PermissionError, FileNotFoundError):
                                    # File might be locked or already deleted, ignore
                                    pass
                            
                            # Display validation results
                            if validation['valid']:
                                st.success("✅ Audio format is optimal for Hume AI analysis!")
                                specs = validation['specs']
                                st.write(f"**Format:** {specs['format']} | **Sample Rate:** {specs['sample_rate']}Hz | **Duration:** {specs['duration']:.2f}s")
                            else:
                                st.warning("⚠️ Audio format can be improved:")
                                for issue in validation.get('issues', []):
                                    st.write(f"• {issue}")
                                
                                if validation.get('recommendations'):
                                    st.write("**Recommendations:**")
                                    for rec in validation['recommendations']:
                                        st.write(f"• {rec}")
                                
                                st.info("Don't worry - the audio will be automatically preprocessed for optimal analysis!")
            
            with col2:
                st.write("")  # Spacing
        
        # Audio preprocessing info
        with st.expander("📋 Audio Preprocessing Info"):
            st.markdown("""
            **Hume AI Audio Requirements:**
            - **Format:** WAV with Linear PCM encoding
            - **Sample Rate:** 16 kHz minimum (higher rates automatically resampled)
            - **Bit Depth:** 16-bit
            - **Channels:** Mono (stereo automatically converted)
            - **Duration:** 3-15 seconds recommended
            - **File Size:** < 10 MB
            
            *Your audio will be automatically preprocessed to meet these specifications.*
            """)
        
        # Process button
        if st.button("🎯 Analyze Audio", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing audio... This may take a moment."):
                try:
                    # Read file content
                    audio_data = uploaded_file.read()
                    
                    # Show detailed preprocessing status
                    preprocessing_placeholder = st.empty()
                    
                    # Check if using Hume AI for enhanced preprocessing info
                    if hasattr(st.session_state.pipeline.emotion_recognizer, '_preprocess_audio_for_hume'):
                        preprocessing_placeholder.info("🔧 Preprocessing audio for Hume AI (converting to 16kHz mono WAV with Linear PCM)...")
                    else:
                        preprocessing_placeholder.info("🔧 Processing audio...")
                    
                    # Process through pipeline (this will automatically preprocess for Hume if needed)
                    result = st.session_state.pipeline.process_audio(audio_data, include_features=True)
                    
                    # Clear preprocessing message
                    preprocessing_placeholder.empty()
                    
                    # Show preprocessing success if Hume was used
                    if hasattr(st.session_state.pipeline.emotion_recognizer, '_preprocess_audio_for_hume'):
                        st.success("✅ Audio automatically preprocessed to meet Hume AI specifications")
                    
                    if result.get('success', False):
                        st.success("✅ Processing completed!")
                        
                        # Display emotion results
                        display_emotion_results(result)
                        
                        # Display transcription and response
                        display_transcription_and_response(result)
                        

                        
                        # Show processing details
                        with st.expander("🔍 Processing Details"):
                            if 'steps' in result:
                                for step_name, step_info in result['steps'].items():
                                    status = "✅" if step_info.get('success', False) else "❌"
                                    duration = step_info.get('duration', 0.0)
                                    st.write(f"{status} **{step_name.replace('_', ' ').title()}**: {duration:.2f}s")
                                    
                                    # Show additional details for emotion recognition
                                    if step_name == 'emotion_recognition' and step_info.get('success', False):
                                        model_type = step_info.get('model_type', 'unknown')
                                        if model_type == 'hume_ai':
                                            st.write("   🎯 **Hume AI Processing**: Audio automatically preprocessed to optimal format")
                                            if 'aggregated_analysis' in step_info:
                                                analysis = step_info['aggregated_analysis']
                                                if 'summary' in analysis:
                                                    summary = analysis['summary']
                                                    st.write(f"   📊 **Emotions Detected**: {summary.get('total_emotions_detected', 0)}")
                                                    st.write(f"   ⚡ **Overall Intensity**: {summary.get('overall_intensity', 0):.3f}")
                                    
                                    if not step_info.get('success', False) and 'error' in step_info:
                                        st.error(f"Error: {step_info['error']}")
                    
                    else:
                        st.error("❌ Processing failed")
                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
                        
                        # Show debug info
                        with st.expander("🔍 Debug Information"):
                            st.json(result)
                
                except Exception as e:
                    st.error(f"❌ Error processing audio: {e}")
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
    

                    sample_rate = 16000
                    duration = 3.0
                    t = np.linspace(0, duration, int(sample_rate * duration))
                    
                    if "Happy" in test_type:
                        audio = 0.3 * np.sin(2 * np.pi * 800 * t) + 0.1 * np.sin(2 * np.pi * 1200 * t)
                    elif "Sad" in test_type:
                        audio = 0.2 * np.sin(2 * np.pi * 200 * t) + 0.05 * np.random.randn(len(t))
                    elif "Neutral" in test_type:
                        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
                    else:  # Complex
                        audio = (0.3 * np.sin(2 * np.pi * 300 * t) + 
                                0.2 * np.sin(2 * np.pi * 600 * t) + 
                                0.1 * np.random.randn(len(t)))
                    
                    # Process the synthetic audio
                    result = st.session_state.pipeline.process_audio(audio)
                    
                    if result.get('success', False):
                        st.success(f"✅ Test completed!")
                        
                        # Display results in columns
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Transcription:** {result.get('transcription', 'N/A')}")
                            st.write(f"**Emotion:** {result.get('emotion', 'N/A')} ({result.get('emotion_confidence', 0.0):.2f})")
                        
                        with col2:
                            response = result.get('response', 'N/A')
                            st.write(f"**Response:** {response[:100]}{'...' if len(response) > 100 else ''}")
                            st.write(f"**Processing Time:** {result.get('processing_time', 0.0):.2f}s")
                    else:
                        st.error("❌ Test failed")
                        if 'error' in result:
                            st.error(result['error'])
                
                except Exception as e:
                    st.error(f"❌ Test error: {e}")
    
    # Conversation history
    if st.session_state.conversation_history:
        st.divider()
        st.subheader("💬 Conversation History")
        
        # Create DataFrame
        df = pd.DataFrame(st.session_state.conversation_history)
        
        # Display recent conversations
        for i, conversation in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5
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
        
        # Download option
        if len(st.session_state.conversation_history) > 0:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Conversation History",
                data=csv,
                file_name=f"emotion_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
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