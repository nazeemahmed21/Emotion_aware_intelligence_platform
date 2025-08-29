"""
Configuration settings for Emotion-Aware Voice Feedback Bot
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# Audio processing settings
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "chunk_size": 1024,
    "channels": 1,
    "format": "wav",
    "max_duration": 30,  # seconds
    "min_duration": 1,   # seconds
}

# Whisper settings
WHISPER_CONFIG = {
    "model_size": "base",  # tiny, base, small, medium, large
    "language": "en",
    "task": "transcribe"
}

# Emotion recognition settings
EMOTION_CONFIG = {
    "confidence_threshold": 0.6,
    "emotions": ["happy", "sad", "angry", "neutral", "excited", "calm", "fearful"],
    "feature_window": 0.050,  # 50ms
    "feature_step": 0.025,    # 25ms
}

# LLM settings
LLM_CONFIG = {
    "model_name": "llama2",
    "fallback_model": "mistral",
    "temperature": 0.7,
    "max_tokens": 150,
    "timeout": 30,  # seconds
}

# Streamlit settings
UI_CONFIG = {
    "page_title": "Emotion-Aware Voice Bot",
    "page_icon": "🎭",
    "layout": "wide",
    "theme": "light"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "emotion_bot.log"
}