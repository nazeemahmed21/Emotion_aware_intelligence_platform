#!/usr/bin/env python3
"""
Test script to verify the audio processing fix
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

def test_audio_processing():
    """Test the audio processing pipeline"""
    print("🧪 Testing Audio Processing Fix")
    print("=" * 40)
    
    try:
        from speech_to_text import WhisperTranscriber
        
        # Create test audio
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create speech-like audio
        audio = (0.3 * np.sin(2 * np.pi * 400 * t) +
                0.2 * np.sin(2 * np.pi * 800 * t) +
                0.1 * np.random.randn(len(t)))
        
        print(f"✅ Created test audio: {len(audio)} samples")
        print(f"   Shape: {audio.shape}, dtype: {audio.dtype}")
        print(f"   Contiguous: {audio.flags['C_CONTIGUOUS']}")
        
        # Initialize transcriber
        print("\n🤖 Loading Whisper...")
        transcriber = WhisperTranscriber(model_size="tiny")
        print("✅ Whisper loaded")
        
        # Test transcription
        print("\n📝 Testing transcription...")
        result = transcriber.transcribe_audio(audio)
        
        if 'error' in result:
            print(f"❌ Transcription failed: {result['error']}")
            return False
        else:
            print(f"✅ Transcription successful!")
            print(f"   Text: '{result.get('text', 'N/A')}'")
            print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_validator():
    """Test the audio validator separately"""
    print("\n🔍 Testing Audio Validator")
    print("=" * 30)
    
    try:
        from audio_validator import validate_audio_for_transcription
        
        # Create test audio
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        print(f"Original audio: shape={audio.shape}, contiguous={audio.flags['C_CONTIGUOUS']}")
        
        # Validate audio
        validated_audio, validation_info = validate_audio_for_transcription(audio, sample_rate)
        
        if validated_audio is not None:
            print(f"✅ Validation successful!")
            print(f"   Validated audio: shape={validated_audio.shape}, contiguous={validated_audio.flags['C_CONTIGUOUS']}")
            print(f"   Quality score: {validation_info.get('quality_score', 0.0):.2f}")
            print(f"   Fixes applied: {validation_info.get('fixes_applied', [])}")
            if validation_info.get('issues'):
                print(f"   Issues: {validation_info['issues']}")
            return True
        else:
            print(f"❌ Validation failed: {validation_info.get('issues', [])}")
            return False
            
    except Exception as e:
        print(f"❌ Validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔧 Audio Processing Fix Tests")
    print("=" * 50)
    
    # Test validator
    validator_ok = test_audio_validator()
    
    # Test full transcription
    transcription_ok = test_audio_processing()
    
    print("\n" + "=" * 50)
    print("📋 Test Results:")
    print(f"   Audio Validator: {'✅ PASS' if validator_ok else '❌ FAIL'}")
    print(f"   Transcription: {'✅ PASS' if transcription_ok else '❌ FAIL'}")
    
    if validator_ok and transcription_ok:
        print("\n🎉 All tests passed! The negative strides issue should be fixed.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return validator_ok and transcription_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)