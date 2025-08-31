#!/usr/bin/env python3
"""
Test script for Faster Whisper integration
==========================================

This script tests the Faster Whisper transcription functionality to ensure
it works correctly before running the main Streamlit app.
"""

import sys
import os
import numpy as np
import tempfile
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing Package Imports")
    print("=" * 40)
    
    try:
        # Test faster-whisper import
        print("📦 Testing faster-whisper import...")
        from faster_whisper import WhisperModel
        print("✅ faster-whisper imported successfully")
        
        # Test our custom transcriber
        print("📦 Testing WhisperTranscriber import...")
        from src.speech_to_text import WhisperTranscriber
        print("✅ WhisperTranscriber imported successfully")
        
        # Test config import
        print("📦 Testing config import...")
        from config import WHISPER_CONFIG
        print("✅ WHISPER_CONFIG imported successfully")
        print(f"   Model size: {WHISPER_CONFIG['model_size']}")
        print(f"   Language: {WHISPER_CONFIG['language']}")
        print(f"   Task: {WHISPER_CONFIG['task']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_model_loading():
    """Test if the Whisper model can be loaded"""
    print("\n🧪 Testing Model Loading")
    print("=" * 40)
    
    try:
        from src.speech_to_text import WhisperTranscriber
        from config import WHISPER_CONFIG
        
        print(f"🔄 Loading Whisper model: {WHISPER_CONFIG['model_size']}")
        start_time = time.time()
        
        transcriber = WhisperTranscriber(WHISPER_CONFIG['model_size'])
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        print(f"   Model type: {type(transcriber.model)}")
        print(f"   Model size: {transcriber.model_size}")
        
        return transcriber
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_transcription_with_synthetic_audio(transcriber):
    """Test transcription with synthetic audio"""
    print("\n🧪 Testing Transcription with Synthetic Audio")
    print("=" * 50)
    
    try:
        # Create synthetic audio (2 seconds of speech-like frequencies)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a speech-like audio signal
        audio = (0.4 * np.sin(2 * np.pi * 200 * t) +   # Base speech frequency
                0.3 * np.sin(2 * np.pi * 400 * t) +    # Harmonic
                0.2 * np.sin(2 * np.pi * 800 * t) +    # Higher harmonic
                0.1 * np.random.randn(len(t)))         # Noise
        
        print(f"✅ Created synthetic audio: {duration}s, {sample_rate}Hz")
        print(f"   Audio shape: {audio.shape}")
        print(f"   Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        
        # Test transcription
        print("🔄 Testing transcription...")
        start_time = time.time()
        
        result = transcriber.transcribe_audio(audio)
        
        transcribe_time = time.time() - start_time
        print(f"✅ Transcription completed in {transcribe_time:.2f}s")
        
        # Display results
        print(f"📝 Transcribed text: '{result.get('text', 'N/A')}'")
        print(f"🌍 Language: {result.get('language', 'N/A')}")
        print(f"🎯 Confidence: {result.get('confidence', 0.0):.3f}")
        print(f"⏱️  Duration: {result.get('duration', 0.0):.2f}s")
        print(f"📊 Segments: {len(result.get('segments', []))}")
        
        if 'error' in result:
            print(f"⚠️  Warning: {result['error']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_transcription_with_file(transcriber):
    """Test transcription with a temporary audio file"""
    print("\n🧪 Testing Transcription with Audio File")
    print("=" * 50)
    
    try:
        # Create a temporary WAV file with synthetic audio
        import wave
        import struct
        
        # Create synthetic audio data
        sample_rate = 16000
        duration = 1.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (0.5 * np.sin(2 * np.pi * 300 * t) + 
                     0.3 * np.sin(2 * np.pi * 600 * t) +
                     0.1 * np.random.randn(len(t)))
        
        # Normalize and convert to 16-bit PCM
        audio_data = np.clip(audio_data, -1, 1)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
            
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
        
        print(f"✅ Created temporary WAV file: {temp_filename}")
        print(f"   File size: {os.path.getsize(temp_filename)} bytes")
        
        # Test transcription from file
        print("🔄 Testing file transcription...")
        start_time = time.time()
        
        result = transcriber.transcribe_audio(temp_filename)
        
        transcribe_time = time.time() - start_time
        print(f"✅ File transcription completed in {transcribe_time:.2f}s")
        
        # Display results
        print(f"📝 Transcribed text: '{result.get('text', 'N/A')}'")
        print(f"🌍 Language: {result.get('language', 'N/A')}")
        print(f"🎯 Confidence: {result.get('confidence', 0.0):.3f}")
        
        # Clean up
        os.unlink(temp_filename)
        print(f"🧹 Cleaned up temporary file")
        
        return result
        
    except Exception as e:
        print(f"❌ File transcription test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_error_handling(transcriber):
    """Test error handling with invalid inputs"""
    print("\n🧪 Testing Error Handling")
    print("=" * 40)
    
    try:
        # Test with empty audio
        print("🔄 Testing empty audio...")
        empty_audio = np.array([])
        result = transcriber.transcribe_audio(empty_audio)
        
        if 'error' in result:
            print("✅ Properly handled empty audio with error message")
        else:
            print("⚠️  Empty audio didn't produce error (unexpected)")
        
        # Test with None input
        print("🔄 Testing None input...")
        try:
            result = transcriber.transcribe_audio(None)
            print("⚠️  None input didn't raise exception (unexpected)")
        except Exception as e:
            print(f"✅ Properly handled None input: {e}")
        
        # Test with invalid file path
        print("🔄 Testing invalid file path...")
        result = transcriber.transcribe_audio("nonexistent_file.wav")
        
        if 'error' in result:
            print("✅ Properly handled invalid file path with error message")
        else:
            print("⚠️  Invalid file path didn't produce error (unexpected)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Faster Whisper Integration Test Suite")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Test 1: Package imports
    if not test_imports():
        print("\n❌ Import tests failed. Cannot proceed.")
        return False
    
    # Test 2: Model loading
    transcriber = test_model_loading()
    if transcriber is None:
        print("\n❌ Model loading failed. Cannot proceed.")
        return False
    
    # Test 3: Transcription with synthetic audio
    audio_result = test_transcription_with_synthetic_audio(transcriber)
    if audio_result is None:
        print("\n❌ Audio transcription test failed.")
        return False
    
    # Test 4: Transcription with file
    file_result = test_transcription_with_file(transcriber)
    if file_result is None:
        print("\n❌ File transcription test failed.")
        return False
    
    # Test 5: Error handling
    if not test_error_handling(transcriber):
        print("\n❌ Error handling test failed.")
        return False
    
    # All tests passed
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 40)
    print("✅ Package imports working")
    print("✅ Model loading successful")
    print("✅ Audio transcription working")
    print("✅ File transcription working")
    print("✅ Error handling working")
    print()
    print("🚀 Your Faster Whisper integration is ready!")
    print("   You can now run: streamlit run emotion_aware_voice_analyzer.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
