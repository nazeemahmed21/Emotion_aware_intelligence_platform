# 🎭 Hume AI Integration

This document explains how to set up and use Hume AI's emotion recognition API with your Emotion-Aware Voice Pipeline.

## 🌟 What is Hume AI?

Hume AI provides state-of-the-art emotion recognition from voice, facial expressions, and text. Their API can detect dozens of nuanced emotions with high accuracy, making it perfect for creating truly empathetic AI applications.

**Key Benefits:**
- ✅ **48+ emotion dimensions** (vs 8 in basic models)
- ✅ **Professional-grade accuracy** from a specialized AI company
- ✅ **Real-time processing** with low latency
- ✅ **Multi-modal analysis** (voice, face, text)
- ✅ **Continuous updates** and improvements

## 🚀 Quick Setup

### 1. Get Your Hume AI Credentials

1. Go to [Hume AI Platform](https://platform.hume.ai/)
2. Sign up for an account
3. Navigate to your dashboard
4. Find your **API Key** and **Secret Key** (if available)

### 2. Set Environment Variables

Create a `.env` file in your project root:

```bash
# Copy from .env.example
cp .env.example .env
```

Edit `.env` and add your credentials:

```bash
# Hume AI Configuration
HUME_API_KEY=your_api_key_here
HUME_SECRET_KEY=your_secret_key_here

# Optional
HUME_WEBHOOK_URL=your_webhook_url_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Test the Integration

```bash
python test_hume_integration.py
```

## 🔧 How It Works

### Integration Priority

The pipeline now uses this priority order for emotion recognition:

1. **Hume AI** (if `HUME_API_KEY` is set) ← **NEW!**
2. **Trained RAVDESS Model** (if available)
3. **Pretrained Model** (fallback)

### Emotion Mapping

Hume AI detects 48+ emotions, which are mapped to the pipeline's standard emotions:

```python
# Hume AI → Pipeline mapping
'Joy' → 'happy'
'Sadness' → 'sad'  
'Anger' → 'angry'
'Fear' → 'fearful'
'Surprise' → 'surprised'
'Calmness' → 'calm'
# ... and many more nuanced emotions
```

### Processing Flow

```mermaid
graph LR
    A[Audio Input] --> B[Save Temp File]
    B --> C[Submit to Hume AI]
    C --> D[Wait for Processing]
    D --> E[Get Predictions]
    E --> F[Map Emotions]
    F --> G[Return Results]
```

## 📊 Usage Examples

### Basic Usage

```python
from src.emotion_recognizer_hume import HumeEmotionRecognizer
import numpy as np

# Initialize
recognizer = HumeEmotionRecognizer()

# Create test audio
sample_rate = 16000
duration = 3.0
t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note

# Predict emotion
result = recognizer.predict_emotion(audio, sample_rate)

if result['success']:
    print(f"Emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"All emotions: {result['all_emotions']}")
```

### Pipeline Integration

```python
from src.pipeline import EmotionAwareVoicePipeline

# Initialize pipeline (automatically uses Hume AI if configured)
pipeline = EmotionAwareVoicePipeline()

# Process audio
result = pipeline.process_audio(audio_data)

print(f"Transcription: {result['transcription']}")
print(f"Emotion: {result['emotion']} ({result['emotion_confidence']:.3f})")
print(f"AI Response: {result['response']}")
```

### Streamlit App

The main Streamlit app automatically uses Hume AI when configured:

```bash
python run.py
```

Upload an audio file and see Hume AI's emotion analysis in action!

## 🎯 Advanced Configuration

### Custom Config File

Create `config/hume_config.json`:

```json
{
  "hume_ai": {
    "api_key": "",
    "secret_key": "",
    "default_granularity": "utterance",
    "score_threshold": 0.6,
    "top_n_emotions": 10,
    "max_retries": 3,
    "job_timeout_seconds": 7200
  }
}
```

### Granularity Levels

Choose how detailed the analysis should be:

- **`word`**: Analyze each word separately
- **`sentence`**: Analyze each sentence
- **`utterance`**: Analyze natural speech segments (default)
- **`turn`**: Analyze conversational turns

### Emotion Thresholds

Configure which emotions to focus on:

```json
{
  "hume_ai": {
    "negative_emotions": [
      "Anger", "Sadness", "Fear", "Disgust", 
      "Anxiety", "Frustration"
    ],
    "intensity_thresholds": {
      "low": 0.3,
      "medium": 0.6, 
      "high": 0.8
    }
  }
}
```

## 🔍 Troubleshooting

### Common Issues

**1. "HUME_API_KEY must be set"**
- Check your `.env` file exists and has the correct key
- Restart your application after setting environment variables

**2. "Job failed" or timeout errors**
- Check your internet connection
- Verify your Hume AI account has API access
- Try with shorter audio files (< 30 seconds for testing)

**3. "Could not import Hume client"**
- Make sure `knowledge_base/hume/` directory exists
- Install missing dependencies: `pip install aiohttp boto3 python-dotenv`

**4. Pipeline not using Hume AI**
- Verify `HUME_API_KEY` is set in environment
- Check the console logs for initialization messages
- Run `test_hume_integration.py` to diagnose

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your code
```

### Test with Real Audio

```bash
# Test with a real audio file
python -c "
from src.emotion_recognizer_hume import HumeEmotionRecognizer
recognizer = HumeEmotionRecognizer()
result = recognizer.predict_emotion('path/to/your/audio.wav')
print(result)
"
```

## 📈 Performance & Costs

### Processing Time
- **Small files** (< 10 seconds): ~30-60 seconds
- **Medium files** (10-60 seconds): ~1-3 minutes  
- **Large files** (> 60 seconds): ~3-10 minutes

### API Limits
- **Rate limits**: Check your Hume AI plan
- **File size**: Up to 100MB per file
- **Duration**: Up to 3 hours per file
- **Concurrent jobs**: Up to 500 jobs

### Cost Optimization
- Use shorter audio clips for testing
- Cache results when possible
- Consider batch processing for multiple files

## 🔒 Security & Privacy

### Data Handling
- Audio files are temporarily uploaded to Hume AI
- Files are processed and then deleted from Hume's servers
- No audio data is permanently stored by Hume AI

### API Keys
- Never commit API keys to version control
- Use environment variables or secure config files
- Rotate keys regularly for production use

### GDPR Compliance
- Hume AI is GDPR compliant
- User consent should be obtained before processing voice data
- Consider data retention policies

## 🚀 Production Deployment

### Environment Setup

```bash
# Production environment variables
export HUME_API_KEY="your_production_key"
export HUME_SECRET_KEY="your_production_secret"
export HUME_WEBHOOK_URL="https://your-app.com/hume-webhook"
```

### Monitoring

Monitor your Hume AI usage:
- Track API calls and costs
- Monitor processing times
- Set up alerts for failures
- Log emotion detection results

### Scaling

For high-volume applications:
- Use webhook callbacks instead of polling
- Implement request queuing
- Consider caching frequent results
- Monitor rate limits

## 🆚 Comparison with Other Models

| Feature | Hume AI | Trained RAVDESS | Pretrained |
|---------|---------|-----------------|------------|
| **Emotions** | 48+ nuanced | 8 basic | 8 basic |
| **Accuracy** | Professional | Good | Basic |
| **Speed** | 30-60s | <1s | <1s |
| **Cost** | API calls | Free | Free |
| **Internet** | Required | Not required | Not required |
| **Updates** | Automatic | Manual | Manual |

## 🎉 Success Stories

### Before Hume AI
```
User: *frustrated voice* "This isn't working!"
Bot: "I detected: neutral (0.6)"
Response: "How can I help you today?"
```

### After Hume AI
```
User: *frustrated voice* "This isn't working!"  
Bot: "I detected: frustrated (0.85), angry (0.72)"
Response: "I can hear the frustration in your voice. Let me help you resolve this issue right away."
```

## 📚 Additional Resources

- [Hume AI Documentation](https://docs.hume.ai/)
- [Hume AI Platform](https://platform.hume.ai/)
- [API Reference](https://docs.hume.ai/reference)
- [Emotion Taxonomy](https://docs.hume.ai/docs/emotion-taxonomy)

## 🤝 Support

Need help with the integration?

1. **Check the logs**: Look for error messages in console output
2. **Run tests**: Use `python test_hume_integration.py`
3. **Check environment**: Verify all environment variables are set
4. **Review docs**: Check Hume AI's official documentation
5. **Contact support**: Reach out to Hume AI support for API issues

---

🎭 **Happy emotion detecting!** Your AI is now powered by professional-grade emotion recognition.