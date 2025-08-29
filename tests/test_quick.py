#!/usr/bin/env python3
"""
Quick test of the emotion pipeline with synthetic audio
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

def test_with_synthetic_audio():
    """Test the pipeline with synthetic audio data"""
    print("🧪 Quick Pipeline Test with Synthetic Audio")
    print("=" * 50)
    
    try:
        from pipeline import EmotionAwareVoicePipeline
        
        # Create synthetic audio (3 seconds of mixed frequencies)
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a complex audio signal that might represent speech
        audio = (0.3 * np.sin(2 * np.pi * 300 * t) +  # Base frequency
                0.2 * np.sin(2 * np.pi * 600 * t) +   # Harmonic
                0.1 * np.sin(2 * np.pi * 1200 * t) +  # Higher harmonic
                0.05 * np.random.randn(len(t)))       # Noise
        
        print(f"✅ Created synthetic audio: {duration}s, {sample_rate}Hz")
        
        # Initialize pipeline
        print("🔄 Initializing pipeline...")
        pipeline = EmotionAwareVoicePipeline()
        
        # Process the audio
        print("🎯 Processing audio...")
        result = pipeline.process_audio(audio, include_features=True)
        
        # Display results
        if result.get('success', False):
            print("\n🎉 SUCCESS! Pipeline is working!")
            print(f"📝 Transcription: '{result.get('transcription', 'N/A')}'")
            print(f"🎭 Emotion: {result.get('emotion', 'N/A')} ({result.get('emotion_confidence', 0.0):.2f})")
            print(f"🤖 Response: '{result.get('response', 'N/A')[:100]}...'")
            print(f"⏱️  Processing time: {result.get('processing_time', 0.0):.2f}s")
            
            # Show step details
            if 'steps' in result:
                print("\n📊 Step Details:")
                for step_name, step_info in result['steps'].items():
                    status = "✅" if step_info.get('success', False) else "❌"
                    duration = step_info.get('duration', 0.0)
                    print(f"  {status} {step_name}: {duration:.2f}s")
                    
                    if not step_info.get('success', False) and 'error' in step_info:
                        print(f"      Error: {step_info['error']}")
        else:
            print("❌ Pipeline failed")
            if 'error' in result:
                print(f"Error: {result['error']}")
            
            # Show step details for debugging
            if 'steps' in result:
                print("\n🔍 Debug Info:")
                for step_name, step_info in result['steps'].items():
                    print(f"  {step_name}: {step_info}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = test_with_synthetic_audio()
    
    if success:
        print("\n🚀 Your setup is working! You can now:")
        print("1. Run the Streamlit app: python run.py")
        print("2. Upload real audio files")
        print("3. Test with the created test files: python tests/test_create_audio.py")
    else:
        print("\n🔧 There are still some issues. Try:")
        print("1. Run: python fix_windows.py")
        print("2. Check that Ollama is running: ollama serve")
        print("3. Verify models are installed: ollama list")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)