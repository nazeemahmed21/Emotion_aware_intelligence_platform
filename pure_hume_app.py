#!/usr/bin/env python3
"""
DEPRECATED: This file has been replaced by emotion_aware_voice_analyzer.py

Please use the new professional application:
streamlit run emotion_aware_voice_analyzer.py
"""

import streamlit as st

st.set_page_config(
    page_title="🚀 Redirecting to New App",
    page_icon="🚀",
    layout="centered"
)

st.markdown("""
# 🚀 Application Upgraded!

This application has been replaced with a professional, enterprise-ready version.

## ✨ New Features
- **Professional UI Design** with modern styling
- **Interactive Visualizations** with Plotly charts
- **Enhanced Analysis** with comprehensive insights
- **Better Performance** with async processing
- **Enterprise Ready** with production-grade code

## 🎯 How to Access the New App

Run this command instead:

```bash
streamlit run emotion_aware_voice_analyzer.py
```

Or click the button below to redirect:
""")

if st.button("🚀 Launch Professional App", type="primary", use_container_width=True):
    st.markdown("""
    <script>
    window.location.href = "http://localhost:8501";
    </script>
    """, unsafe_allow_html=True)
    
st.markdown("""
---
**Note**: The old `pure_hume_app.py` has been deprecated in favor of the new professional application.
""")

# Check setup
api_key = os.getenv('HUME_API_KEY')
if not api_key:
    st.error("❌ HUME_API_KEY not found in environment variables")
    st.info("Please add HUME_API_KEY=your_key_here to your .env file")
    st.stop()

st.success(f"✅ HUME_API_KEY configured: {api_key[:10]}...")

# Try to import Hume client
try:
    from hume.hume_client import HumeClient, HumeConfig, GranularityLevel
    st.success("✅ Hume client imported successfully")
except Exception as e:
    st.error(f"❌ Cannot import Hume client: {e}")
    st.stop()

def analyze_with_hume(audio_file_path):
    """Analyze audio with Hume AI"""
    try:
        # Create config and client
        config = HumeConfig(
            api_key=api_key,
            secret_key=os.getenv('HUME_SECRET_KEY'),
            webhook_url=os.getenv('HUME_WEBHOOK_URL')
        )
        client = HumeClient(config)
        granularity = GranularityLevel.UTTERANCE
        
        # Run async analysis
        async def run_analysis():
            # Submit job
            job_id = await client.submit_files([audio_file_path], granularity)
            st.info(f"📤 Job submitted: {job_id}")
            
            # Wait for completion
            success = await client.wait_for_job(job_id)
            
            if success:
                # Get predictions
                predictions = await client.get_job_predictions(job_id, format="json")
                return predictions
            else:
                return None
        
        # Run the analysis
        return asyncio.run(run_analysis())
        
    except Exception as e:
        st.error(f"❌ Analysis error: {e}")
        return None

def extract_emotions(predictions):
    """Extract emotions from Hume predictions"""
    all_emotions = []
    
    try:
        # Handle list response format (as confirmed by direct test)
        if isinstance(predictions, list):
            predictions_list = predictions
        else:
            st.error("❌ Unexpected prediction format")
            return []
        
        # Process predictions
        for file_prediction in predictions_list:
            # Handle nested structure
            if "results" in file_prediction:
                file_predictions = file_prediction["results"].get("predictions", [])
            else:
                file_predictions = [file_prediction]
            
            for pred in file_predictions:
                models = pred.get("models", {})
                
                # Process each model
                for model_name in ["prosody", "burst", "language"]:
                    if model_name not in models:
                        continue
                    
                    model_data = models[model_name]
                    grouped_predictions = model_data.get("grouped_predictions", [])
                    
                    for group in grouped_predictions:
                        predictions_inner = group.get("predictions", [])
                        
                        for prediction in predictions_inner:
                            if "emotions" in prediction:
                                emotions = prediction["emotions"]
                                
                                for emotion in emotions:
                                    all_emotions.append({
                                        'name': emotion.get('name', ''),
                                        'score': emotion.get('score', 0.0),
                                        'model': model_name
                                    })
        
        return all_emotions
        
    except Exception as e:
        st.error(f"❌ Error extracting emotions: {e}")
        return []

def display_results(emotions):
    """Display analysis results"""
    if not emotions:
        st.warning("⚠️ No emotions detected")
        return
    
    st.success(f"✅ Detected {len(emotions)} emotions across all models")
    
    # Group by emotion name and calculate stats
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
    
    # Calculate averages and sort
    for name, stats in emotion_stats.items():
        stats['mean_score'] = stats['total_score'] / stats['count']
        stats['max_score'] = max(stats['scores'])
        stats['unique_models'] = list(set(stats['models']))
    
    # Sort by mean score
    sorted_emotions = sorted(emotion_stats.items(), key=lambda x: x[1]['mean_score'], reverse=True)
    
    # Display top 10 emotions
    st.subheader("🏆 Top 10 Emotions (Pure Hume Results)")
    
    top_10 = sorted_emotions[:10]
    emotion_data = []
    
    for i, (name, stats) in enumerate(top_10):
        emotion_data.append({
            'Rank': i + 1,
            'Emotion': name,
            'Mean Score': f"{stats['mean_score']:.3f}",
            'Peak Score': f"{stats['max_score']:.3f}",
            'Frequency': stats['count'],
            'Models': ', '.join(stats['unique_models'])
        })
    
    df = pd.DataFrame(emotion_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Hesitancy analysis
    st.subheader("🤔 Hesitancy Analysis")
    
    hesitancy_keywords = ['anxiety', 'nervousness', 'uncertainty', 'confusion', 'hesitation', 'doubt', 'worry']
    hesitancy_emotions = []
    
    for name, stats in emotion_stats.items():
        if any(keyword in name.lower() for keyword in hesitancy_keywords):
            hesitancy_emotions.append((name, stats['mean_score']))
    
    if hesitancy_emotions:
        hesitancy_emotions.sort(key=lambda x: x[1], reverse=True)
        avg_hesitancy = sum(score for _, score in hesitancy_emotions) / len(hesitancy_emotions)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Hesitancy", f"{avg_hesitancy:.3f}")
        with col2:
            st.metric("Hesitancy Indicators", len(hesitancy_emotions))
        
        st.write("**Hesitancy Emotions Detected:**")
        for name, score in hesitancy_emotions:
            st.write(f"• {name}: {score:.3f}")
    else:
        st.info("No hesitancy indicators detected")
    
    # Model breakdown
    st.subheader("🧠 Model Breakdown")
    
    model_stats = {'prosody': [], 'burst': [], 'language': []}
    for emotion in emotions:
        model_stats[emotion['model']].append(emotion['score'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prosody_scores = model_stats['prosody']
        st.markdown("**🎵 Prosody Model**")
        if prosody_scores:
            st.write(f"Emotions: {len(prosody_scores)}")
            st.write(f"Avg Score: {np.mean(prosody_scores):.3f}")
            st.write(f"Max Score: {max(prosody_scores):.3f}")
        else:
            st.write("No emotions detected")
    
    with col2:
        burst_scores = model_stats['burst']
        st.markdown("**⚡ Burst Model**")
        if burst_scores:
            st.write(f"Emotions: {len(burst_scores)}")
            st.write(f"Avg Score: {np.mean(burst_scores):.3f}")
            st.write(f"Max Score: {max(burst_scores):.3f}")
        else:
            st.write("No emotions detected")
    
    with col3:
        language_scores = model_stats['language']
        st.markdown("**📝 Language Model**")
        if language_scores:
            st.write(f"Emotions: {len(language_scores)}")
            st.write(f"Avg Score: {np.mean(language_scores):.3f}")
            st.write(f"Max Score: {max(language_scores):.3f}")
        else:
            st.write("No emotions detected")

# File upload
st.subheader("📁 Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
    help="Upload an audio file for Hume emotion analysis"
)

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.success(f"✅ Uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
    
    # Audio player
    st.audio(uploaded_file.read(), format='audio/wav')
    uploaded_file.seek(0)  # Reset file pointer
    
    # Analyze button
    if st.button("🎯 Analyze with Pure Hume AI", type="primary", use_container_width=True):
        with st.spinner("🔄 Processing with Hume AI... This may take 30-60 seconds..."):
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
                    st.success("✅ Pure Hume analysis completed!")
                    
                    # Extract and display emotions
                    emotions = extract_emotions(predictions)
                    display_results(emotions)
                    
                    # Show raw data in expander
                    with st.expander("🔍 Raw Hume AI Response"):
                        st.json(predictions)
                else:
                    st.error("❌ Analysis failed")
            
            except Exception as e:
                st.error(f"❌ Error processing audio: {e}")
                import traceback
                st.code(traceback.format_exc())

# Instructions
with st.expander("📋 Instructions"):
    st.markdown("""
    **How to use:**
    1. Upload an audio file (WAV, MP3, M4A, OGG, FLAC)
    2. Click "Analyze with Pure Hume AI"
    3. Wait 30-60 seconds for processing
    4. View pure Hume results with:
       - Top 10 emotions ranked by intensity
       - Hesitancy analysis
       - Model breakdown (Prosody, Burst, Language)
    
    **Requirements:**
    - HUME_API_KEY must be set in .env file
    - Audio should contain clear speech for best results
    - Processing is done entirely by Hume AI (no local models)
    """)

st.markdown("---")
st.info("This app uses your existing knowledge_base/hume implementation for direct Hume AI analysis.")