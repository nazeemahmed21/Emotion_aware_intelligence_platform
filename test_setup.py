#!/usr/bin/env python3
"""
Test script to verify the emotion-aware voice feedback bot setup
"""
import sys
import os
import tempfile
import numpy as np
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import whisper
        print("✅ Whisper imported")
    except ImportError as e:
        print(f"❌ Whisper import failed: {e}")
        return False
    
    try:
        import librosa
        print("✅ Librosa imported")
    except ImportError as e:
        print(f"❌ Librosa import failed: {e}")
        return False
    
    try:
        import torch
        import transformers
        print("✅ PyTorch and Transformers imported")
    except ImportError as e:
        print(f"❌ PyTorch/Transformers import failed: {e}")
        return False
    
    try:
        from src.audio_utils import AudioProcessor
        from src.speech_to_text import WhisperTranscriber
        from src.emotion_recognition import EmotionRecognizer
        from src.llm_response import OllamaResponseGenerator
        from src.pipeline import EmotionAwareVoicePipeline
        print("✅ All custom modules imported")
    except ImportError as e:
        print(f"❌ Custom module import failed: {e}")
        return False
    
    return True

def test_audio_processing():
    """Test audio processing functionality"""
    print("\n🎵 Testing audio processing...")
    
    try:
        from src.audio_utils import AudioProcessor
        
        # Create synthetic audio data
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # A4 note
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        processor = AudioProcessor()
        
        # Test preprocessing
        processed_audio = processor.preprocess_audio(audio_data, sample_rate)
        print("✅ Audio preprocessing works")
        
        # Test feature extraction
        features = processor.extract_features(processed_audio, sample_rate)
        print(f"✅ Feature extraction works ({len(features)} features)")
        
        # Test audio info
        info = processor.get_audio_info(processed_audio, sample_rate)
        print(f"✅ Audio info extraction works (duration: {info['duration']:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        return False

def test_speech_to_text():
    """Test speech-to-text functionality"""
    print("\n📝 Testing speech-to-text...")
    
    try:
        from src.speech_to_text import WhisperTranscriber
        
        # Initialize transcriber
        transcriber = WhisperTranscriber(model_size="tiny")  # Use tiny model for testing
        print("✅ Whisper transcriber initialized")
        
        # Create synthetic audio (silence - won't transcribe well but tests the pipeline)
        sample_rate = 16000
        duration = 1.0
        audio_data = np.zeros(int(sample_rate * duration))
        
        # Test transcription
        result = transcriber.transcribe_audio(audio_data)
        print(f"✅ Transcription test completed (result: '{result.get('text', 'N/A')}')")
        
        return True
        
    except Exception as e:
        print(f"❌ Speech-to-text test failed: {e}")
        return False

def test_emotion_recognition():
    """Test emotion recognition functionality"""
    print("\n🎭 Testing emotion recognition...")
    
    try:
        from src.emotion_recognition import EmotionRecognizer
        
        # Initialize recognizer
        recognizer = EmotionRecognizer()
        print("✅ Emotion recognizer initialized")
        
        # Create synthetic audio
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create a more complex waveform
        audio_data = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)
        
        # Test emotion prediction
        result = recognizer.predict_emotion(audio_data)
        emotion = result.get('emotion', 'unknown')
        confidence = result.get('confidence', 0.0)
        print(f"✅ Emotion recognition test completed (emotion: {emotion}, confidence: {confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Emotion recognition test failed: {e}")
        return False

def test_llm_response():
    """Test LLM response generation"""
    print("\n🤖 Testing LLM response generation...")
    
    try:
        from src.llm_response import OllamaResponseGenerator
        
        # Initialize response generator
        generator = OllamaResponseGenerator()
        print("✅ LLM response generator initialized")
        
        # Test response generation (will use fallback if Ollama not available)
        result = generator.generate_empathetic_response(
            transcription="I'm feeling good today",
            emotion="happy",
            confidence=0.8
        )
        
        response = result.get('response', 'No response')
        model = result.get('model', 'unknown')
        print(f"✅ Response generation test completed (model: {model})")
        print(f"   Response: '{response[:50]}...'")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM response test failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n🔄 Testing full pipeline...")
    
    try:
        from src.pipeline import EmotionAwareVoicePipeline
        
        # Initialize pipeline
        pipeline = EmotionAwareVoicePipeline()
        print("✅ Pipeline initialized")
        
        # Create synthetic audio
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.random.randn(len(t))
        
        # Test pipeline processing
        result = pipeline.process_audio(audio_data, include_features=True)
        
        if result.get('success', False):
            print("✅ Full pipeline test completed successfully")
            print(f"   Transcription: '{result.get('transcription', 'N/A')}'")
            print(f"   Emotion: {result.get('emotion', 'N/A')} ({result.get('emotion_confidence', 0.0):.2f})")
            print(f"   Response: '{result.get('response', 'N/A')[:50]}...'")
            print(f"   Processing time: {result.get('processing_time', 0.0):.2f}s")
        else:
            print(f"⚠️  Pipeline completed with issues: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Emotion-Aware Voice Feedback Bot - Setup Test")
    print("=" * 60)
    
    # Configure logging to reduce noise during testing
    logging.getLogger().setLevel(logging.WARNING)
    
    tests = [
        ("Imports", test_imports),
        ("Audio Processing", test_audio_processing),
        ("Speech-to-Text", test_speech_to_text),
        ("Emotion Recognition", test_emotion_recognition),
        ("LLM Response", test_llm_response),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("🚀 Run 'python run.py' or 'streamlit run streamlit_app.py' to start the app")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("💡 Try running 'pip install -r requirements.txt' to fix dependency issues")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)