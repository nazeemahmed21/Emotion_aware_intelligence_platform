8. Testing Strategy Documentation
Overview
A comprehensive testing strategy ensures your Emotion-Aware Voice Bot is reliable, performs well, and maintains quality throughout development. This documentation covers unit tests, integration tests, performance tests, and test data management.

Test Architecture
Test Directory Structure
text
tests/
├── unit/
│   ├── test_audio_processing.py
│   ├── test_emotion_detection.py
│   ├── test_whisper_integration.py
│   └── test_ollama_responses.py
├── integration/
│   ├── test_full_pipeline.py
│   ├── test_api_endpoints.py
│   └── test_websocket_communication.py
├── performance/
│   ├── test_load_performance.py
│   ├── test_latency.py
│   └── test_memory_usage.py
├── fixtures/
│   ├── sample_audio/
│   ├── mock_responses/
│   └── test_data.py
└── conftest.py
Unit Tests
Audio Processing Unit Tests
python
# tests/unit/test_audio_processing.py
import pytest
import numpy as np
from unittest.mock import Mock, patch
from audio_processing import extract_audio_features, preprocess_audio

class TestAudioProcessing:
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data for testing"""
        sample_rate = 16000
        duration = 3.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        return audio_data, sample_rate
    
    def test_extract_audio_features(self, sample_audio_data):
        """Test feature extraction returns expected features"""
        audio_data, sample_rate = sample_audio_data
        features = extract_audio_features(audio_data, sample_rate)
        
        assert isinstance(features, dict)
        assert 'mfcc_mean' in features
        assert 'spectral_centroid' in features
        assert 'zcr' in features
        assert all(isinstance(v, (int, float)) for v in features.values())
    
    def test_preprocess_audio(self, sample_audio_data):
        """Test audio preprocessing maintains data integrity"""
        audio_data, sample_rate = sample_audio_data
        processed_audio = preprocess_audio(audio_data, sample_rate)
        
        assert len(processed_audio) == len(audio_data)
        assert isinstance(processed_audio, np.ndarray)
        assert processed_audio.dtype == np.float32
    
    @patch('audio_processing.librosa.load')
    def test_feature_extraction_error_handling(self, mock_load):
        """Test error handling in feature extraction"""
        mock_load.side_effect = Exception("Load error")
        
        with pytest.raises(Exception) as exc_info:
            extract_audio_features("invalid_path.wav")
        assert "Load error" in str(exc_info.value)
Emotion Detection Unit Tests
python
# tests/unit/test_emotion_detection.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from emotion_detection import detect_emotion, EmotionDetector

class TestEmotionDetection:
    
    @pytest.fixture
    def sample_features(self):
        """Sample audio features for testing"""
        return {
            'mfcc_mean': 0.5,
            'spectral_centroid': 1500,
            'zcr': 0.12,
            'rms': 0.8
        }
    
    @pytest.fixture
    def mock_model(self):
        """Mock sklearn model for testing"""
        mock_model = MagicMock()
        mock_model.predict.return_value = ['happy']
        mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])
        return mock_model
    
    def test_detect_emotion(self, sample_features, mock_model):
        """Test emotion detection with mock model"""
        with patch('emotion_detection.load_model') as mock_load:
            mock_load.return_value = mock_model
            
            emotion, confidence = detect_emotion(sample_features)
            
            assert emotion == 'happy'
            assert 0 <= confidence <= 1
            mock_model.predict.assert_called_once()
    
    def test_emotion_detector_class(self, sample_features, mock_model):
        """Test EmotionDetector class functionality"""
        detector = EmotionDetector()
        detector.model = mock_model
        
        result = detector.detect(sample_features)
        
        assert result['emotion'] == 'happy'
        assert 'confidence' in result
        assert 'features' in result
    
    @pytest.mark.parametrize("input_features,expected", [
        ({}, ("neutral", 0.5)),  # Empty features
        (None, ("neutral", 0.5)),  # None input
    ])
    def test_edge_cases(self, input_features, expected, mock_model):
        """Test edge cases in emotion detection"""
        with patch('emotion_detection.load_model') as mock_load:
            mock_load.return_value = mock_model
            
            emotion, confidence = detect_emotion(input_features)
            assert emotion == expected[0]
Whisper Integration Tests
python
# tests/unit/test_whisper_integration.py
import pytest
from unittest.mock import patch, MagicMock
from whisper_transcription import transcribe_audio, WhisperTranscriber

class TestWhisperIntegration:
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model"""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'text': 'This is a test transcription',
            'segments': [{'text': 'This is a test transcription'}]
        }
        return mock_model
    
    def test_transcribe_audio(self, mock_whisper_model):
        """Test audio transcription with mock Whisper"""
        with patch('whisper_transcription.whisper.load_model') as mock_load:
            mock_load.return_value = mock_whisper_model
            
            result = transcribe_audio("test_audio.wav")
            
            assert result == 'This is a test transcription'
            mock_whisper_model.transcribe.assert_called_once()
    
    @patch('whisper_transcription.whisper.load_model')
    def test_transcriber_class(self, mock_load, mock_whisper_model):
        """Test WhisperTranscriber class"""
        mock_load.return_value = mock_whisper_model
        transcriber = WhisperTranscriber(model_size="base")
        
        result = transcriber.transcribe("test_audio.wav")
        
        assert result['text'] == 'This is a test transcription'
        assert 'segments' in result
    
    def test_transcription_error_handling(self):
        """Test error handling in transcription"""
        with patch('whisper_transcription.whisper.load_model') as mock_load:
            mock_load.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception) as exc_info:
                transcribe_audio("test_audio.wav")
            assert "Model loading failed" in str(exc_info.value)
Ollama Response Tests
python
# tests/unit/test_ollama_responses.py
import pytest
from unittest.mock import patch, AsyncMock
from ollama_response import generate_empathetic_response, EmotionAwareResponder

class TestOllamaResponses:
    
    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama response"""
        return {
            'message': {'content': 'I understand you are feeling happy!'},
            'response': 'I understand you are feeling happy!'
        }
    
    @pytest.mark.asyncio
    async def test_generate_empathetic_response(self, mock_ollama_response):
        """Test response generation with mock Ollama"""
        with patch('ollama_response.ollama.chat', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_ollama_response
            
            response = await generate_empathetic_response(
                "I'm feeling great!", "happy", 0.9
            )
            
            assert "happy" in response.lower()
            mock_chat.assert_called_once()
    
    @pytest.mark.parametrize("emotion,expected_keywords", [
        ("happy", ["great", "wonderful", "happy"]),
        ("sad", ["sorry", "support", "here for you"]),
        ("angry", ["understand", "frustrat", "calm"]),
    ])
    @pytest.mark.asyncio
    async def test_emotion_specific_responses(self, emotion, expected_keywords, mock_ollama_response):
        """Test emotion-specific response patterns"""
        with patch('ollama_response.ollama.chat', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {'message': {'content': f"Response for {emotion}"}}
            
            response = await generate_empathetic_response("test", emotion, 0.8)
            
            # For actual implementation, you'd check for emotion-specific keywords
            assert emotion in response or any(kw in response.lower() for kw in expected_keywords)
    
    def test_fallback_responses(self):
        """Test fallback responses when Ollama fails"""
        responder = EmotionAwareResponder()
        
        # Test fallback for each emotion
        fallback = responder.get_fallback_response("happy")
        assert "glad" in fallback.lower()
        
        fallback = responder.get_fallback_response("sad")
        assert "sorry" in fallback.lower()
Integration Tests
Full Pipeline Integration Test
python
# tests/integration/test_full_pipeline.py
import pytest
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock
from audio_processing import extract_audio_features
from emotion_detection import detect_emotion
from whisper_transcription import transcribe_audio
from ollama_response import generate_empathetic_response

class TestFullPipeline:
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate realistic sample audio data"""
        np.random.seed(42)  # For reproducible tests
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create audio with multiple frequency components
        audio_data = (
            0.3 * np.sin(2 * np.pi * 200 * t) +  # Base frequency
            0.2 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency
            0.1 * np.sin(2 * np.pi * 2000 * t)   # High frequency
        )
        return audio_data, sample_rate
    
    @pytest.mark.integration
    def test_audio_processing_pipeline(self, sample_audio_data, tmp_path):
        """Test complete audio processing pipeline"""
        audio_data, sample_rate = sample_audio_data
        
        # Save audio to temporary file
        audio_file = tmp_path / "test_audio.wav"
        import scipy.io.wavfile as wavfile
        wavfile.write(audio_file, sample_rate, audio_data)
        
        # Test feature extraction
        features = extract_audio_features(str(audio_file))
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Test emotion detection (with mock model)
        with patch('emotion_detection.load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = ['happy']
            mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])
            mock_load.return_value = mock_model
            
            emotion, confidence = detect_emotion(features)
            assert emotion == 'happy'
            assert 0.7 <= confidence <= 1.0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_integration(self, sample_audio_data, tmp_path):
        """Test complete integration with mocks"""
        audio_data, sample_rate = sample_audio_data
        audio_file = tmp_path / "test_audio.wav"
        
        import scipy.io.wavfile as wavfile
        wavfile.write(audio_file, sample_rate, audio_data)
        
        # Mock all external dependencies
        with patch('whisper_transcription.whisper.load_model') as mock_whisper, \
             patch('emotion_detection.load_model') as mock_emotion, \
             patch('ollama_response.ollama.chat', new_callable=AsyncMock) as mock_ollama:
            
            # Setup mocks
            mock_whisper_model = MagicMock()
            mock_whisper_model.transcribe.return_value = {'text': 'I feel happy today'}
            mock_whisper.return_value = mock_whisper_model
            
            mock_emotion_model = MagicMock()
            mock_emotion_model.predict.return_value = ['happy']
            mock_emotion_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])
            mock_emotion.return_value = mock_emotion_model
            
            mock_ollama.return_value = {'message': {'content': 'I am glad you are happy!'}}
            
            # Run pipeline
            transcription = transcribe_audio(str(audio_file))
            features = extract_audio_features(str(audio_file))
            emotion, confidence = detect_emotion(features)
            response = await generate_empathetic_response(transcription, emotion, confidence)
            
            # Verify results
            assert transcription == 'I feel happy today'
            assert emotion == 'happy'
            assert confidence == 0.8
            assert 'happy' in response.lower()
API Endpoint Tests
python
# tests/integration/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app

class TestAPIEndpoints:
    
    @pytest.fixture
    def client(self):
        """Test client for FastAPI"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_audio_file(self, tmp_path):
        """Create sample audio file for testing"""
        audio_file = tmp_path / "test_audio.wav"
        
        # Create a simple WAV file
        import wave
        import struct
        
        with wave.open(str(audio_file), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            
            # Generate some sample data
            for i in range(1000):
                value = int(32767 * 0.5 * (i % 100) / 100)
                data = struct.pack('<h', value)
                wav_file.writeframes(data)
        
        return audio_file
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @patch('main.transcribe_audio')
    @patch('main.extract_audio_features')
    @patch('main.detect_emotion')
    @patch('main.generate_empathetic_response', new_callable=AsyncMock)
    def test_analyze_audio_endpoint(self, mock_response, mock_emotion, 
                                  mock_features, mock_transcribe, client, sample_audio_file):
        """Test audio analysis endpoint"""
        # Setup mocks
        mock_transcribe.return_value = "Test transcription"
        mock_features.return_value = {"feature1": 0.5}
        mock_emotion.return_value = ("happy", 0.85)
        mock_response.return_value = "I'm glad you're happy!"
        
        with open(sample_audio_file, "rb") as f:
            response = client.post(
                "/api/analyze-audio",
                files={"file": ("test_audio.wav", f, "audio/wav")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["transcription"] == "Test transcription"
        assert data["emotion"] == "happy"
        assert data["confidence"] == 0.85
    
    def test_conversations_endpoint(self, client):
        """Test conversations endpoint"""
        response = client.get("/api/conversations")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    @patch('main.ConversationManager')
    def test_websocket_connection(self, mock_manager, client):
        """Test WebSocket connection"""
        # This would require more complex testing with async WebSocket client
        # For now, test that the endpoint exists
        with client.websocket_connect("/ws/audio-analysis") as websocket:
            # Basic connection test
            assert websocket.client_state.name == "CONNECTED"
Performance Tests
Load Testing
python
# tests/performance/test_load_performance.py
import pytest
import asyncio
from locust import HttpUser, task, between
from unittest.mock import patch

class EmotionBotUser(HttpUser):
    wait_time = between(1, 5)
    
    @task
    def analyze_audio(self):
        # Mock audio file upload
        files = {"file": ("test_audio.wav", b"fake_audio_data", "audio/wav")}
        
        with patch('main.transcribe_audio') as mock_transcribe, \
             patch('main.extract_audio_features') as mock_features, \
             patch('main.detect_emotion') as mock_emotion, \
             patch('main.generate_empathetic_response') as mock_response:
            
            mock_transcribe.return_value = "Test message"
            mock_features.return_value = {}
            mock_emotion.return_value = ("neutral", 0.5)
            mock_response.return_value = "Test response"
            
            self.client.post("/api/analyze-audio", files=files)
    
    @task(3)
    def get_health(self):
        self.client.get("/health")
    
    @task(2)
    def get_conversations(self):
        self.client.get("/api/conversations")

class TestLoadPerformance:
    
    @pytest.mark.performance
    def test_concurrent_requests(self):
        """Test system under concurrent load"""
        # This would typically run with locust or similar load testing framework
        # For unit testing, we can test the performance metrics collection
        
        from utils.monitoring import REQUEST_LATENCY, PROCESSING_TIME
        
        # Simulate some requests
        with PROCESSING_TIME.labels(stage="audio_processing").time():
            import time
            time.sleep(0.1)  # Simulate processing time
        
        # Check that metrics are being collected
        assert REQUEST_LATENCY._metrics != {}
        assert PROCESSING_TIME._metrics != {}
Latency Testing
python
# tests/performance/test_latency.py
import pytest
import time
from unittest.mock import patch
from audio_processing import extract_audio_features
from emotion_detection import detect_emotion

class TestLatency:
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio for latency testing"""
        import numpy as np
        sample_rate = 16000
        duration = 5.0  # Longer duration for meaningful timing
        t = np.linspace(0, duration, int(sample_rate * duration))
        return 0.5 * np.sin(2 * np.pi * 440 * t), sample_rate
    
    @pytest.mark.performance
    def test_feature_extraction_latency(self, sample_audio_data):
        """Test feature extraction latency"""
        audio_data, sample_rate = sample_audio_data
        
        start_time = time.time()
        features = extract_audio_features(audio_data, sample_rate)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process 5 seconds of audio in reasonable time
        assert processing_time < 2.0  # Less than 2 seconds
        assert len(features) > 0  # Should extract features
    
    @pytest.mark.performance
    def test_emotion_detection_latency(self):
        """Test emotion detection latency"""
        sample_features = {
            'mfcc_mean': 0.5,
            'spectral_centroid': 1500,
            'zcr': 0.12,
            'rms': 0.8
        }
        
        with patch('emotion_detection.load_model') as mock_load:
            mock_model = type('MockModel', (), {
                'predict': lambda self, x: ['happy'],
                'predict_proba': lambda self, x: [[0.1, 0.8, 0.1]]
            })()
            mock_load.return_value = mock_model
            
            start_time = time.time()
            emotion, confidence = detect_emotion(sample_features)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            assert processing_time < 0.1  # Should be very fast
            assert emotion == 'happy'
Test Fixtures and Data
Test Data Management
python
# tests/fixtures/test_data.py
import pytest
import numpy as np
import os
from pathlib import Path

@pytest.fixture(scope="session")
def test_audio_dir():
    """Directory containing test audio files"""
    return Path(__file__).parent / "sample_audio"

@pytest.fixture
def happy_audio_data(test_audio_dir):
    """Audio data that should be classified as happy"""
    audio_file = test_audio_dir / "happy_sample.wav"
    if audio_file.exists():
        import librosa
        return librosa.load(audio_file, sr=16000)
    else:
        # Generate synthetic happy audio (bright, energetic)
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = (
            0.4 * np.sin(2 * np.pi * 300 * t) +
            0.3 * np.sin(2 * np.pi * 600 * t) +
            0.2 * np.sin(2 * np.pi * 1200 * t) +
            0.1 * np.sin(2 * np.pi * 2400 * t)
        )
        return audio, sr

@pytest.fixture
def sad_audio_data(test_audio_dir):
    """Audio data that should be classified as sad"""
    audio_file = test_audio_dir / "sad_sample.wav"
    if audio_file.exists():
        import librosa
        return librosa.load(audio_file, sr=16000)
    else:
        # Generate synthetic sad audio (dark, slow)
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = (
            0.5 * np.sin(2 * np.pi * 200 * t) +
            0.3 * np.sin(2 * np.pi * 400 * t) +
            0.1 * np.sin(2 * np.pi * 800 * t)
        )
        return audio, sr

@pytest.fixture
def mock_ollama_responses():
    """Mock responses for different emotions"""
    return {
        "happy": "I'm so glad to hear you're feeling happy! That's wonderful!",
        "sad": "I'm sorry you're feeling down. I'm here to listen and support you.",
        "angry": "I understand you're feeling frustrated. Let's work through this together.",
        "neutral": "Thank you for sharing. How can I assist you today?",
    }
Configuration for Testing
python
# tests/conftest.py
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import os

# Add source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def mock_external_services():
    """Auto-mock external services for all tests"""
    with patch('whisper_transcription.whisper.load_model') as mock_whisper, \
         patch('emotion_detection.load_model') as mock_emotion, \
         patch('ollama_response.ollama.chat', new_callable=AsyncMock) as mock_ollama:
        
        # Setup default mocks
        mock_whisper_model = MagicMock()
        mock_whisper_model.transcribe.return_value = {'text': 'Test transcription'}
        mock_whisper.return_value = mock_whisper_model
        
        mock_emotion_model = MagicMock()
        mock_emotion_model.predict.return_value = ['neutral']
        mock_emotion_model.predict_proba.return_value = [[0.2, 0.6, 0.2]]
        mock_emotion.return_value = mock_emotion_model
        
        mock_ollama.return_value = {'message': {'content': 'Test response'}}
        
        yield {
            'whisper': mock_whisper,
            'emotion': mock_emotion,
            'ollama': mock_ollama
        }
Running Tests
Test Configuration
python
# pytest.ini
[pytest]
testpaths = tests
asyncio_mode = auto
filterwarnings =
    ignore::DeprecationWarning
    ignore::UserWarning
Test Commands
bash
# Run all tests
pytest tests/ -v

# Run specific test types
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v --durations=0

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_audio_processing.py -v

# Run tests with specific marker
pytest tests/ -m "integration" -v
GitHub Actions Test Workflow
yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:alpine
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
This comprehensive testing strategy ensures your Emotion-Aware Voice Bot is thoroughly tested at all levels, from individual units to full integration, with proper performance monitoring and continuous integration support.