# 🎭 Professional Emotion-Aware Voice Bot

A modern, industry-standard web application for real-time emotion recognition and empathetic AI responses. Built with Streamlit, featuring professional UI/UX, real-time voice recording, and advanced AI pipeline integration.

## ✨ Features

### 🎙️ **Real-time Voice Recording**
- **Browser-based recording** with multiple fallback options
- **WebRTC integration** for real-time audio capture
- **File upload support** for maximum compatibility
- **Audio visualization** with waveform display

### 🎭 **Advanced Emotion Recognition**
- **Multi-model approach** with Wav2Vec2 and rule-based fallbacks
- **Real-time confidence scoring** with visual feedback
- **7 emotion categories**: Happy, Sad, Angry, Neutral, Calm, Excited, Fearful
- **Confidence gauges** and emotion distribution analytics

### 🤖 **Intelligent Response Generation**
- **Local LLM integration** via Ollama (LLaMA 2, Mistral)
- **Context-aware responses** based on detected emotions
- **Conversation history** with emotional context
- **Fallback responses** for offline operation

### 📊 **Professional Analytics**
- **Real-time dashboards** with emotion trends
- **Performance metrics** and processing statistics
- **Conversation insights** and pattern analysis
- **Export capabilities** for data analysis

### 🎨 **Modern UI/UX**
- **Industry-standard design** with clean, professional interface
- **Responsive layout** optimized for desktop and mobile
- **Real-time visual feedback** during processing
- **Accessibility compliant** with proper contrast and navigation

## 🚀 Quick Start

### **Option 1: Automated Installation (Recommended)**

```bash
# Clone or download the project
cd emotion-aware-voice-bot

# Run automated installation
python install_professional.py

# Start the application
python run_professional.py
```

### **Option 2: Manual Installation**

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install and start Ollama
# Download from https://ollama.ai/download
ollama serve
ollama pull llama2

# 5. Run the application
streamlit run professional_app.py
```

## 🏗️ Architecture

### **Modern Component Structure**
```
professional_app.py          # Main application entry point
├── src/
│   ├── ui_components.py     # Modern UI component library
│   ├── app_state.py         # Centralized state management
│   ├── enhanced_recorder.py # Multi-method audio recording
│   ├── audio_utils.py       # Audio processing utilities
│   ├── speech_to_text.py    # Whisper integration
│   ├── emotion_recognition.py # Emotion AI models
│   ├── llm_response.py      # Ollama LLM integration
│   └── pipeline.py          # Main processing pipeline
├── install_professional.py  # Automated installer
├── run_professional.py     # Professional runner
└── requirements.txt         # Dependencies
```

### **Technology Stack**
- **Frontend**: Streamlit with custom CSS/HTML
- **Audio Processing**: Librosa, SoundFile, WebRTC
- **Speech-to-Text**: OpenAI Whisper
- **Emotion Recognition**: Hugging Face Transformers (Wav2Vec2)
- **Response Generation**: Ollama (LLaMA 2, Mistral)
- **Visualization**: Plotly, custom components
- **State Management**: Streamlit session state with custom wrapper

## 🎯 Usage Guide

### **1. Voice Recording**
- **Real-time Recording**: Click the microphone button to start/stop recording
- **File Upload**: Drag and drop or select audio files (WAV, MP3, M4A)
- **Recording Tips**: Speak clearly for 2-10 seconds, minimize background noise

### **2. Emotion Analysis**
- **Automatic Processing**: Audio is processed immediately after recording
- **Visual Feedback**: Real-time confidence gauges and emotion displays
- **Multiple Emotions**: View all detected emotions with confidence scores

### **3. AI Responses**
- **Contextual Responses**: AI generates empathetic responses based on detected emotion
- **Conversation History**: All interactions are saved with emotional context
- **Response Customization**: Adjust response style in settings

### **4. Analytics Dashboard**
- **Emotion Trends**: Track emotional patterns over time
- **Performance Metrics**: Monitor processing speed and accuracy
- **Export Data**: Download conversation history as JSON/CSV

## 🔧 Configuration

### **Audio Recording Methods**
The application automatically detects and uses the best available recording method:

1. **audio-recorder-streamlit** (Preferred)
   - Browser-based real-time recording
   - Best compatibility and user experience

2. **streamlit-webrtc** (Advanced)
   - WebRTC-based recording
   - Requires HTTPS in production

3. **File Upload** (Fallback)
   - Always available
   - Supports multiple audio formats

### **Emotion Recognition Models**
- **Primary**: Wav2Vec2 (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
- **Fallback**: Rule-based detection using audio features
- **Confidence Thresholds**: Configurable per emotion type

### **LLM Configuration**
- **Primary Model**: LLaMA 2 (7B parameters)
- **Fallback Model**: Mistral (7B parameters)
- **Local Processing**: All inference happens locally via Ollama
- **Response Styles**: Empathetic, Professional, Friendly, Concise

## 📊 Performance

### **Processing Times** (typical hardware)
- **Audio Loading**: < 0.1s
- **Speech-to-Text**: 1-3s (depends on audio length)
- **Emotion Recognition**: 0.5-1s
- **Response Generation**: 2-8s (depends on model and hardware)
- **Total Pipeline**: 3-12s

### **System Requirements**
- **Minimum**: 4GB RAM, 2GB free disk space
- **Recommended**: 8GB RAM, 4GB free disk space
- **GPU**: Optional (CPU inference supported)
- **Network**: Required for initial model downloads

## 🔒 Privacy & Security

### **Data Privacy**
- **Local Processing**: All AI models run locally on your machine
- **No Data Upload**: Audio and conversations never leave your device
- **Session Storage**: Data stored only in browser session (not persistent)
- **Optional History**: Conversation history can be disabled in settings

### **Security Features**
- **Open Source Models**: All AI models are open source and auditable
- **No External APIs**: No data sent to third-party services
- **Secure by Design**: Minimal attack surface with local processing

## 🛠️ Development

### **Project Structure**
```
emotion-aware-voice-bot/
├── professional_app.py          # Main Streamlit application
├── src/                         # Source code modules
│   ├── ui_components.py         # UI component library
│   ├── app_state.py            # State management
│   ├── enhanced_recorder.py    # Audio recording
│   ├── audio_utils.py          # Audio processing
│   ├── speech_to_text.py       # Whisper integration
│   ├── emotion_recognition.py  # Emotion AI
│   ├── llm_response.py         # LLM integration
│   └── pipeline.py             # Processing pipeline
├── install_professional.py     # Automated installer
├── run_professional.py        # Application runner
├── requirements.txt            # Python dependencies
├── .streamlit/                 # Streamlit configuration
├── data/                       # Data directory
├── models/                     # Model cache
└── logs/                       # Application logs
```

### **Code Quality Standards**
- **Type Hints**: All functions have proper type annotations
- **Error Handling**: Comprehensive exception handling with user feedback
- **Logging**: Structured logging for debugging and monitoring
- **Documentation**: Clear docstrings and inline comments
- **Modularity**: Clean separation of concerns across modules

### **Testing**
```bash
# Run installation tests
python install_professional.py

# Test individual components
python test_whisper_direct.py

# Test full pipeline
python quick_test.py
```

## 🚀 Deployment

### **Local Development**
```bash
streamlit run professional_app.py
```

### **Production Deployment**
```bash
# Using Docker (create Dockerfile)
docker build -t emotion-voice-bot .
docker run -p 8501:8501 emotion-voice-bot

# Using cloud platforms
# - Streamlit Cloud: Connect GitHub repository
# - Heroku: Add buildpacks for Python and FFmpeg
# - AWS/GCP: Use container deployment
```

### **Environment Variables**
```bash
# Optional configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
OLLAMA_HOST=http://localhost:11434
```

## 🤝 Contributing

### **Development Setup**
```bash
# Clone repository
git clone <repository-url>
cd emotion-aware-voice-bot

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black src/ *.py

# Type checking
mypy src/
```

### **Contributing Guidelines**
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Follow** code quality standards (type hints, documentation)
4. **Test** your changes thoroughly
5. **Submit** a pull request with clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for Whisper speech-to-text model
- **Hugging Face** for emotion recognition models and transformers library
- **Ollama** for local LLM inference capabilities
- **Streamlit** for the excellent web application framework
- **Meta** for LLaMA models and PyTorch
- **Google** for material design inspiration

## 📞 Support

### **Getting Help**
1. **Check the troubleshooting section** in this README
2. **Run the diagnostic tools**: `python install_professional.py`
3. **Check the logs** in the `logs/` directory
4. **Create an issue** on GitHub with:
   - Error messages
   - System information (OS, Python version)
   - Steps to reproduce

### **Common Issues**
- **Audio recording not working**: Check browser permissions and try file upload
- **Ollama connection failed**: Ensure Ollama is running (`ollama serve`)
- **Model loading errors**: Check available disk space and internet connection
- **Performance issues**: Try smaller models or reduce audio quality

---

**Built with ❤️ for natural human-AI interaction**

🎭 **Emotion-Aware Voice Bot** - Where technology meets empathy