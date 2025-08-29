#!/usr/bin/env python3
"""
Simple test for transcription functionality
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

def test_simple_transcription():
    """Test basic transcription without complex processing"""
    print("🧪 Testing Simple Transcription")
    print("=" * 40)
    
    try:
        import whisper
        
        # Load tiny model for testing
        print("Loading Whisper tiny model...")
        model = whisper.load_model("tiny")
        print("✅ Model loaded")
        
        # Create simple test audio
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Simple sine wave
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        # Ensure proper format
        audio = np.ascontiguousarray(audio.astype(np.float32))
        
        print(f"Test audio: shape={audio.shape}, dtype={audio.dtype}, contiguous={audio.flags['C_CONTIGUOUS']}")
        
        # Test transcription
        print("Testing transcription...")
        result = model.transcribe(audio)
        
        print(f"✅ Transcription successful!")
        print(f"   Text: '{result['text']}'")
        print(f"   Language: {result.get('language', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_transcription():
    """Test transcription with a real audio file"""
    print("\n🎵 Testing File Transcription")
    print("=" * 35)
    
    try:
        # Create a test WAV file
        import soundfile as sf
        
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create more complex audio
        audio = (0.3 * np.sin(2 * np.pi * 300 * t) +
                0.2 * np.sin(2 * np.pi * 600 * t) +
                0.1 * np.random.randn(len(t)))
        
        # Save to file
        test_file = "test_audio_temp.wav"
        sf.write(test_file, audio, sample_rate)
        print(f"✅ Created test file: {test_file}")
        
        # Test our transcription module
        from speech_to_text import WhisperTranscriber
        
        transcriber = WhisperTranscriber(model_size="tiny")
        result = transcriber.transcribe_audio(test_file)
        
        if 'error' in result:
            print(f"❌ Transcription failed: {result['error']}")
            success = False
        else:
            print(f"✅ Transcription successful!")
            print(f"   Text: '{result.get('text', 'N/A')}'")
            print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
            success = True
        
        # Clean up
        try:
            os.remove(test_file)
        except:
            pass
        
        return success
        
    except Exception as e:
        print(f"❌ File test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run transcription tests"""
    print("🔧 Simple Transcription Tests")
    print("=" * 50)
    
    # Test basic transcription
    basic_ok = test_simple_transcription()
    
    # Test file transcription
    file_ok = test_file_transcription()
    
    print("\n" + "=" * 50)
    print("📋 Test Results:")
    print(f"   Basic Transcription: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"   File Transcription: {'✅ PASS' if file_ok else '❌ FAIL'}")
    
    if basic_ok and file_ok:
        print("\n🎉 All tests passed! Transcription should work in the app.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return basic_ok and file_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)