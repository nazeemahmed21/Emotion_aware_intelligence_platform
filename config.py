#!/usr/bin/env python3
"""
Configuration Management for Emotion-Aware Voice Intelligence Platform
======================================================================

Centralized configuration management with environment variable support,
validation, and type safety for production deployment.

Author: Emotion AI Team
Version: 2.0.0
License: MIT
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    supported_formats: List[str]
    max_file_size_mb: int
    sample_rate: int
    pause_threshold: float
    recording_color: str
    neutral_color: str
    
    @classmethod
    def from_env(cls) -> 'AudioConfig':
        """Create AudioConfig from environment variables"""
        return cls(
            supported_formats=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
            max_file_size_mb=int(os.getenv('MAX_FILE_SIZE_MB', '200')),
            sample_rate=int(os.getenv('SAMPLE_RATE', '44100')),
            pause_threshold=float(os.getenv('PAUSE_THRESHOLD', '2.0')),
            recording_color=os.getenv('RECORDING_COLOR', '#ff6b6b'),
            neutral_color=os.getenv('NEUTRAL_COLOR', '#667eea')
        )

@dataclass
class HumeConfig:
    """Hume AI API configuration"""
    api_key: str
    secret_key: Optional[str]
    webhook_url: Optional[str]
    timeout_seconds: int
    max_retries: int
    
    @classmethod
    def from_env(cls) -> 'HumeConfig':
        """Create HumeConfig from environment variables"""
        api_key = os.getenv('HUME_API_KEY')
        if not api_key:
            raise ValueError("HUME_API_KEY environment variable is required")
        
        return cls(
            api_key=api_key,
            secret_key=os.getenv('HUME_SECRET_KEY'),
            webhook_url=os.getenv('HUME_WEBHOOK_URL'),
            timeout_seconds=int(os.getenv('HUME_TIMEOUT_SECONDS', '300')),
            max_retries=int(os.getenv('HUME_MAX_RETRIES', '3'))
        )

@dataclass
class UIConfig:
    """User interface configuration"""
    page_title: str
    page_icon: str
    layout: str
    initial_sidebar_state: str
    theme_primary_color: str
    theme_background_color: str
    theme_secondary_background_color: str
    theme_text_color: str
    
    @classmethod
    def from_env(cls) -> 'UIConfig':
        """Create UIConfig from environment variables"""
        return cls(
            page_title=os.getenv('PAGE_TITLE', 'Emotion-Aware Voice Intelligence Platform'),
            page_icon=os.getenv('PAGE_ICON', '🧠'),
            layout=os.getenv('LAYOUT', 'wide'),
            initial_sidebar_state=os.getenv('INITIAL_SIDEBAR_STATE', 'expanded'),
            theme_primary_color=os.getenv('THEME_PRIMARY_COLOR', '#667eea'),
            theme_background_color=os.getenv('THEME_BACKGROUND_COLOR', '#ffffff'),
            theme_secondary_background_color=os.getenv('THEME_SECONDARY_BG_COLOR', '#f0f2f6'),
            theme_text_color=os.getenv('THEME_TEXT_COLOR', '#262730')
        )

@dataclass
class AnalysisConfig:
    """Analysis configuration"""
    analysis_depths: List[str]
    default_analysis_depth: str
    hesitancy_keywords: List[str]
    confidence_threshold: float
    
    @classmethod
    def from_env(cls) -> 'AnalysisConfig':
        """Create AnalysisConfig from environment variables"""
        return cls(
            analysis_depths=["Standard", "Comprehensive", "Research Grade"],
            default_analysis_depth=os.getenv('DEFAULT_ANALYSIS_DEPTH', 'Comprehensive'),
            hesitancy_keywords=[
                'anxiety', 'nervousness', 'uncertainty', 'confusion',
                'hesitation', 'doubt', 'worry', 'stress', 'awkwardness'
            ],
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
        )

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str
    format: str
    log_file: str
    max_file_size_mb: int
    backup_count: int
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create LoggingConfig from environment variables"""
        return cls(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            log_file=os.getenv('LOG_FILE', 'logs/emotion_analyzer.log'),
            max_file_size_mb=int(os.getenv('LOG_MAX_FILE_SIZE_MB', '10')),
            backup_count=int(os.getenv('LOG_BACKUP_COUNT', '5'))
        )

@dataclass
class WhisperConfig:
    """Whisper transcription configuration"""
    model_size: str
    language: str
    task: str
    temperature: float
    compression_ratio_threshold: float
    logprob_threshold: float
    no_speech_threshold: float
    
    @classmethod
    def from_env(cls) -> 'WhisperConfig':
        """Create WhisperConfig from environment variables"""
        return cls(
            model_size=os.getenv('WHISPER_MODEL_SIZE', 'base'),
            language=os.getenv('WHISPER_LANGUAGE', 'en'),
            task=os.getenv('WHISPER_TASK', 'transcribe'),
            temperature=float(os.getenv('WHISPER_TEMPERATURE', '0.0')),
            compression_ratio_threshold=float(os.getenv('WHISPER_COMPRESSION_RATIO_THRESHOLD', '2.4')),
            logprob_threshold=float(os.getenv('WHISPER_LOGPROB_THRESHOLD', '-1.0')),
            no_speech_threshold=float(os.getenv('WHISPER_NO_SPEECH_THRESHOLD', '0.6'))
        )

class ApplicationConfig:
    """Main application configuration class"""
    
    def __init__(self):
        """Initialize configuration from environment variables"""
        try:
            self.audio = AudioConfig.from_env()
            self.hume = HumeConfig.from_env()
            self.ui = UIConfig.from_env()
            self.analysis = AnalysisConfig.from_env()
            self.logging = LoggingConfig.from_env()
            self.whisper = WhisperConfig.from_env()
            
            # Ensure required directories exist
            self._create_directories()
            
            # Validate configuration
            self._validate_config()
            
            logger.info("Application configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _create_directories(self) -> None:
        """Create required directories if they don't exist"""
        directories = [
            Path(self.logging.log_file).parent,
            Path('data'),
            Path('exports'),
            Path('temp')
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self) -> None:
        """Validate configuration values"""
        # Validate audio configuration
        if self.audio.max_file_size_mb <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be positive")
        
        if self.audio.sample_rate <= 0:
            raise ValueError("SAMPLE_RATE must be positive")
        
        # Validate Hume configuration
        if len(self.hume.api_key) < 10:
            raise ValueError("HUME_API_KEY appears to be invalid")
        
        # Validate analysis configuration
        if self.analysis.confidence_threshold < 0 or self.analysis.confidence_threshold > 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit page configuration"""
        return {
            'page_title': self.ui.page_title,
            'page_icon': self.ui.page_icon,
            'layout': self.ui.layout,
            'initial_sidebar_state': self.ui.initial_sidebar_state
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'level': getattr(logging, self.logging.level.upper()),
            'format': self.logging.format,
            'filename': self.logging.log_file
        }
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return os.getenv('ENVIRONMENT', 'production').lower() == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.is_development()
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for debugging"""
        return {
            'environment': os.getenv('ENVIRONMENT', 'production'),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'config_loaded': True,
            'hume_api_configured': bool(self.hume.api_key),
            'log_level': self.logging.level
        }

# Global configuration instance
try:
    config = ApplicationConfig()
    # Create WHISPER_CONFIG for backward compatibility
    WHISPER_CONFIG = {
        "model_size": config.whisper.model_size,
        "language": config.whisper.language,
        "task": config.whisper.task,
        "temperature": config.whisper.temperature,
        "compression_ratio_threshold": config.whisper.compression_ratio_threshold,
        "logprob_threshold": config.whisper.logprob_threshold,
        "no_speech_threshold": config.whisper.no_speech_threshold
    }
except Exception as e:
    logger.error(f"Failed to initialize application configuration: {e}")
    # Create a minimal config for error handling
    config = None
    # Create minimal WHISPER_CONFIG for backward compatibility
    WHISPER_CONFIG = {
        "model_size": "base",
        "language": "en",
        "task": "transcribe",
        "temperature": 0.0,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6
    }

def get_config() -> ApplicationConfig:
    """Get the global configuration instance"""
    if config is None:
        raise RuntimeError("Application configuration not initialized")
    return config

def validate_environment() -> bool:
    """Validate that the environment is properly configured"""
    try:
        required_vars = ['HUME_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False
        
        # Try to create configuration
        test_config = ApplicationConfig()
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

if __name__ == "__main__":
    """Configuration testing and validation"""
    print("🔧 Configuration Validation")
    print("=" * 40)
    
    try:
        config = ApplicationConfig()
        print("✅ Configuration loaded successfully")
        
        env_info = config.get_environment_info()
        for key, value in env_info.items():
            print(f"   {key}: {value}")
        
        print("\n📊 Configuration Summary:")
        print(f"   Audio formats: {', '.join(config.audio.supported_formats)}")
        print(f"   Max file size: {config.audio.max_file_size_mb}MB")
        print(f"   Sample rate: {config.audio.sample_rate}Hz")
        print(f"   Analysis depth: {config.analysis.default_analysis_depth}")
        print(f"   Log level: {config.logging.level}")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        exit(1)