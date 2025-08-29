#!/usr/bin/env python3
"""
Test Hume AI Integration with the Pipeline
"""
import sys
import os
import numpy as np
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.append('src')
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_hume_environment():
    """Test if Hume AI environment is set up correctly"""
    print("🔧 Testing Hume AI Environment Setup")
    print("=" * 45)
    
    # Check environment variables
    api_key = os.getenv('HUME_API_KEY')
    secret_key = os.getenv('HUME_SECRET_KEY')
    
    if api_key:
        print(f"✅ HUME_API_KEY is set (length: {len(api_key)})")
    else:
        print("❌ HUME_API_KEY is not set")
        print("   Please set it in your .env file or environment")
        return False
    
    if secret_key:
        print(f"✅ HUME_SECRET_KEY is set (length: {len(secret_key)})")
    else:
        print("⚠️  HUME_SECRET_KEY is not set (optional)")
    
    return True

def test_hume_recognizer():
    """Test the Hume emotion recognizer directly"""
    print("\n🤖 Testing Hume Emotion Recognizer")
    print("=" * 40)
    
    try:
        from emotion_recognizer_hume import HumeEmotionRecognizer
        
        # Initialize recognizer
        print("Initializing Hume AI recognizer...")
        recognizer = HumeEmotionRecognizer()
        print("✅ Hume AI recognizer initialized")
        
        # Show model info
        info = recognizer.get_model_info()
        print(f"Model: {info['model_name']}")
        print(f"Type: {info['model_type']}")
        print(f"Supported emotions: {len(info['supported_emotions'])} emotions")
        
        # Test with synthetic audio
        print("\n🎵 Testing with synthetic audio...")
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create happy-sounding audio (higher frequencies)
        audio = 0.3 * np.sin(2 * np.pi * 800 * t) + 0.1 * np.sin(2 * np.pi * 1200 * t)
        
        print("Sending audio to Hume AI (this may take 30-60 seconds)...")
        result = recognizer.predict_emotion(audio, sample_rate)
        
        if result['success']:
            print("✅ Hume AI prediction successful!")
            print(f"   Primary emotion: {result['emotion']} ({result['confidence']:.3f})")
            print(f"   Job ID: {result.get('job_id', 'N/A')}")
            
            print("\n📊 Top 5 emotion scores:")
            for i, emotion_data in enumerate(result['all_emotions'][:5]):
                print(f"   {i+1}. {emotion_data['emotion']}: {emotion_data['confidence']:.3f}")
            
            return True
        else:
            print(f"❌ Hume AI prediction failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Hume recognizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test the full pipeline with Hume AI"""
    print("\n🔄 Testing Pipeline Integration")
    print("=" * 35)
    
    try:
        from pipeline import EmotionAwareVoicePipeline
        
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = EmotionAwareVoicePipeline()
        print("✅ Pipeline initialized")
        
        # Check which emotion recognizer is being used
        recognizer_type = getattr(pipeline.emotion_recognizer, 'get_model_info', lambda: {'model_type': 'unknown'})()
        print(f"Emotion recognizer: {recognizer_type.get('model_type', 'unknown')}")
        
        if recognizer_type.get('model_type') == 'hume_ai':
            print("✅ Pipeline is using Hume AI emotion recognition")
        else:
            print("⚠️  Pipeline is not using Hume AI (check environment variables)")
            return False
        
        # Test with synthetic audio
        print("\n🎵 Testing full pipeline...")
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create more complex audio
        audio = (0.3 * np.sin(2 * np.pi * 400 * t) +
                0.2 * np.sin(2 * np.pi * 800 * t) +
                0.1 * np.random.randn(len(t)))
        
        print("Processing through full pipeline (this may take 1-2 minutes)...")
        result = pipeline.process_audio(audio, include_features=True)
        
        if result.get('success', False):
            print("✅ Full pipeline test successful!")
            print(f"   Transcription: '{result.get('transcription', 'N/A')}'")
            print(f"   Emotion: {result.get('emotion', 'N/A')} ({result.get('emotion_confidence', 0.0):.3f})")
            print(f"   Response: '{result.get('response', 'N/A')[:100]}...'")
            print(f"   Processing time: {result.get('processing_time', 0.0):.2f}s")
            
            # Show step details
            if 'steps' in result:
                print("\n📊 Step breakdown:")
                for step_name, step_info in result['steps'].items():
                    status = "✅" if step_info.get('success', False) else "❌"
                    duration = step_info.get('duration', 0.0)
                    print(f"   {status} {step_name.replace('_', ' ').title()}: {duration:.2f}s")
            
            return True
        else:
            print(f"❌ Pipeline test failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Hume AI integration tests"""
    print("🎭 Hume AI Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_hume_environment),
        ("Hume Recognizer", test_hume_recognizer),
        ("Pipeline Integration", test_pipeline_integration),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} ERROR: {e}")
    
    print(f"\n🎯 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Hume AI integration is working correctly.")
        print("\n🚀 Next steps:")
        print("1. Run the main app: python run.py")
        print("2. Upload audio files to test emotion recognition")
        print("3. Check that emotions are detected using Hume AI")
    else:
        print("\n⚠️  Some tests failed. Please check:")
        print("1. HUME_API_KEY is set in your .env file")
        print("2. Your Hume AI account has API access")
        print("3. Internet connection is working")
        print("4. All dependencies are installed")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)