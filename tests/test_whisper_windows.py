#!/usr/bin/env python3
"""
Windows-specific test for Whisper transcription
"""
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

def test_whisper_with_file():
    """Test Whisper with a real audio file"""
    print("🎤 Testing Whisper with Audio File")
    print("=" * 40)
    
    try:
        from speech_to_text import WhisperTranscriber
        import soundfile as sf
        
        # Create a simple test audio file
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a simple tone (this won't transcribe to words, but tests the pipeline)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        # Save to a real file
        test_file = "test_audio_temp.wav"
        sf.write(test_file, audio, sample_rate)
        print(f"✅ Created test file: {test_file}")
        
        # Test transcription
        transcriber = WhisperTranscriber(model_size="tiny")  # Use tiny for speed
        print("✅ Whisper transcriber loaded")
        
        # Test with file path
        result = transcriber.transcribe_audio(test_file)
        print(f"✅ File transcription result: '{result.get('text', 'N/A')}'")
        
        # Test with numpy array
        result2 = transcriber.transcribe_audio(audio)
        print(f"✅ Array transcription result: '{result2.get('text', 'N/A')}'")
        
        # Test with bytes
        with open(test_file, 'rb') as f:
            audio_bytes = f.read()
        result3 = transcriber.transcribe_audio(audio_bytes)
        print(f"✅ Bytes transcription result: '{result3.get('text', 'N/A')}'")
        
        # Clean up
        try:
            os.remove(test_file)
            print("✅ Cleaned up test file")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_whisper_with_speech():
    """Test Whisper with synthesized speech-like audio"""
    print("\n🗣️ Testing Whisper with Speech-like Audio")
    print("=" * 45)
    
    try:
        from speech_to_text import WhisperTranscriber
        import soundfile as sf
        
        # Create more speech-like audio (multiple frequencies)
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Simulate formants of speech
        f1 = 700  # First formant
        f2 = 1220  # Second formant
        f3 = 2600  # Third formant
        
        audio = (0.3 * np.sin(2 * np.pi * f1 * t) +
                0.2 * np.sin(2 * np.pi * f2 * t) +
                0.1 * np.sin(2 * np.pi * f3 * t) +
                0.05 * np.random.randn(len(t)))
        
        # Add some amplitude modulation to simulate speech rhythm
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
        audio = audio * envelope
        
        # Save to file
        test_file = "test_speech_temp.wav"
        sf.write(test_file, audio, sample_rate)
        print(f"✅ Created speech-like test file: {test_file}")
        
        # Test transcription
        transcriber = WhisperTranscriber(model_size="tiny")
        result = transcriber.transcribe_audio(test_file)
        
        print(f"📝 Transcription: '{result.get('text', 'N/A')}'")
        print(f"🔍 Language: {result.get('language', 'N/A')}")
        print(f"📊 Confidence: {result.get('confidence', 0.0):.2f}")
        
        # Clean up
        try:
            os.remove(test_file)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Speech test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline_with_file():
    """Test the full pipeline with a file"""
    print("\n🔄 Testing Full Pipeline with File")
    print("=" * 35)
    
    try:
        from pipeline import EmotionAwareVoicePipeline
        import soundfile as sf
        
        # Create test audio
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.random.randn(len(t))
        
        # Save to file
        test_file = "pipeline_test_temp.wav"
        sf.write(test_file, audio, sample_rate)
        
        # Test pipeline
        pipeline = EmotionAwareVoicePipeline()
        result = pipeline.process_audio(test_file)
        
        if result.get('success', False):
            print("✅ Pipeline succeeded!")
            print(f"📝 Transcription: '{result.get('transcription', 'N/A')}'")
            print(f"🎭 Emotion: {result.get('emotion', 'N/A')}")
            print(f"🤖 Response: '{result.get('response', 'N/A')[:50]}...'")
        else:
            print("❌ Pipeline failed")
            if 'error' in result:
                print(f"Error: {result['error']}")
        
        # Clean up
        try:
            os.remove(test_file)
        except:
            pass
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Windows-specific tests"""
    print("🪟 Windows Whisper Tests")
    print("=" * 25)
    
    tests = [
        ("Whisper File Test", test_whisper_with_file),
        ("Whisper Speech Test", test_whisper_with_speech),
        ("Full Pipeline Test", test_full_pipeline_with_file),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All Windows tests passed!")
        print("🚀 Your setup should work with the Streamlit app now")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)