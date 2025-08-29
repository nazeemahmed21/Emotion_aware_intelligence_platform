#!/usr/bin/env python3
"""
Direct test of Whisper without temporary files
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

def test_whisper_direct():
    """Test Whisper with direct numpy array input"""
    print("🎤 Testing Whisper Direct (No Temp Files)")
    print("=" * 45)
    
    try:
        from speech_to_text import WhisperTranscriber
        
        # Initialize transcriber
        print("Loading Whisper model...")
        transcriber = WhisperTranscriber(model_size="tiny")  # Use tiny for speed
        print("✅ Whisper model loaded")
        
        # Test 1: Simple sine wave
        print("\n📊 Test 1: Simple sine wave")
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio1 = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
        
        result1 = transcriber.transcribe_audio(audio1)
        print(f"Result: '{result1.get('text', 'N/A')}'")
        print(f"Confidence: {result1.get('confidence', 0.0):.2f}")
        
        # Test 2: More complex audio (speech-like)
        print("\n📊 Test 2: Speech-like audio")
        # Create formant-like frequencies
        f1, f2, f3 = 700, 1220, 2600  # Typical formants for vowel sounds
        audio2 = (0.3 * np.sin(2 * np.pi * f1 * t) +
                 0.2 * np.sin(2 * np.pi * f2 * t) +
                 0.1 * np.sin(2 * np.pi * f3 * t))
        
        # Add amplitude modulation (like speech rhythm)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        audio2 = audio2 * envelope
        
        result2 = transcriber.transcribe_audio(audio2)
        print(f"Result: '{result2.get('text', 'N/A')}'")
        print(f"Confidence: {result2.get('confidence', 0.0):.2f}")
        
        # Test 3: Noisy audio
        print("\n📊 Test 3: Noisy audio")
        audio3 = 0.2 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.random.randn(len(t))
        
        result3 = transcriber.transcribe_audio(audio3)
        print(f"Result: '{result3.get('text', 'N/A')}'")
        print(f"Confidence: {result3.get('confidence', 0.0):.2f}")
        
        # Test 4: Silence (should return empty or minimal text)
        print("\n📊 Test 4: Silence")
        audio4 = np.zeros(int(sample_rate * 1.0))  # 1 second of silence
        
        result4 = transcriber.transcribe_audio(audio4)
        print(f"Result: '{result4.get('text', 'N/A')}'")
        print(f"Confidence: {result4.get('confidence', 0.0):.2f}")
        
        print("\n✅ All Whisper direct tests completed!")
        print("Note: Results may be empty or gibberish since we're using synthetic audio")
        print("The important thing is that no file errors occurred!")
        
        return True
        
    except Exception as e:
        print(f"❌ Whisper direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline_direct():
    """Test the full pipeline with direct numpy input"""
    print("\n🔄 Testing Full Pipeline Direct")
    print("=" * 35)
    
    try:
        from pipeline import EmotionAwareVoicePipeline
        
        # Create test audio
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create more interesting audio
        audio = (0.3 * np.sin(2 * np.pi * 400 * t) +      # Base frequency
                0.2 * np.sin(2 * np.pi * 800 * t) +       # Harmonic
                0.1 * np.sin(2 * np.pi * 1200 * t) +      # Higher harmonic
                0.05 * np.random.randn(len(t)))           # Noise
        
        # Add some variation to make it more "emotional"
        envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
        audio = audio * envelope
        
        print("✅ Created test audio")
        
        # Initialize pipeline
        print("Loading pipeline...")
        pipeline = EmotionAwareVoicePipeline()
        print("✅ Pipeline loaded")
        
        # Process audio
        print("Processing audio...")
        result = pipeline.process_audio(audio, include_features=True)
        
        if result.get('success', False):
            print("\n🎉 SUCCESS! Full pipeline working!")
            print(f"📝 Transcription: '{result.get('transcription', 'N/A')}'")
            print(f"🎭 Emotion: {result.get('emotion', 'N/A')} ({result.get('emotion_confidence', 0.0):.2f})")
            print(f"🤖 Response: '{result.get('response', 'N/A')[:100]}...'")
            print(f"⏱️  Total time: {result.get('processing_time', 0.0):.2f}s")
            
            # Show step breakdown
            print("\n📊 Step Breakdown:")
            if 'steps' in result:
                for step_name, step_info in result['steps'].items():
                    status = "✅" if step_info.get('success', False) else "❌"
                    duration = step_info.get('duration', 0.0)
                    print(f"  {status} {step_name.replace('_', ' ').title()}: {duration:.2f}s")
                    
                    if not step_info.get('success', False) and 'error' in step_info:
                        print(f"      ⚠️  {step_info['error']}")
            
            return True
        else:
            print("❌ Pipeline failed")
            if 'error' in result:
                print(f"Error: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run direct tests without temporary files"""
    print("🚀 Direct Audio Tests (Windows Compatible)")
    print("=" * 50)
    
    tests = [
        ("Whisper Direct", test_whisper_direct),
        ("Full Pipeline Direct", test_full_pipeline_direct),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
    
    print(f"\n🎯 Final Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All direct tests passed!")
        print("🚀 Your setup should work perfectly now!")
        print("\nNext steps:")
        print("1. Run: python run.py")
        print("2. Upload audio files and test!")
    else:
        print("\n⚠️  Some tests still failed.")
        print("But if Whisper Direct passed, the main issue is fixed!")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)