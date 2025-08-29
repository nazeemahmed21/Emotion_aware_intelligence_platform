#!/usr/bin/env python3
"""
Windows-specific fixes for the Emotion-Aware Voice Feedback Bot
"""
import os
import sys
import subprocess
import tempfile

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed")
            return True
        else:
            print("❌ FFmpeg not working properly")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not found")
        print("Install FFmpeg:")
        print("1. Download from https://ffmpeg.org/download.html")
        print("2. Or use chocolatey: choco install ffmpeg")
        print("3. Or use winget: winget install FFmpeg")
        return False

def fix_temp_directory():
    """Ensure temp directory is writable"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp_file:
            tmp_file.write(b'test')
            tmp_file.flush()
        print("✅ Temporary directory is writable")
        return True
    except Exception as e:
        print(f"❌ Temporary directory issue: {e}")
        return False

def check_ollama_models():
    """Check and fix Ollama model names"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is working")
            print("Available models:")
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip():
                    model_name = line.split()[0]
                    print(f"  - {model_name}")
            
            # Check if we have the required models
            models = result.stdout
            if 'llama2' in models or 'mistral' in models:
                print("✅ Required models are available")
                return True
            else:
                print("⚠️  No suitable models found")
                print("Run: ollama pull llama2")
                return False
        else:
            print("❌ Ollama not working")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found")
        print("Install from: https://ollama.ai/download")
        return False

def create_test_audio():
    """Create a test audio file for testing"""
    try:
        import numpy as np
        import soundfile as sf
        
        # Create a simple test audio (1 second of sine wave)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Save test audio
        os.makedirs('data', exist_ok=True)
        test_file = 'data/test_audio.wav'
        sf.write(test_file, audio, sample_rate)
        
        print(f"✅ Created test audio file: {test_file}")
        return test_file
    except Exception as e:
        print(f"❌ Failed to create test audio: {e}")
        return None

def test_pipeline_components():
    """Test individual pipeline components"""
    print("\n🧪 Testing pipeline components...")
    
    try:
        # Test audio processing
        from src.audio_utils import AudioProcessor
        processor = AudioProcessor()
        print("✅ Audio processor loaded")
        
        # Test with synthetic audio
        import numpy as np
        sample_rate = 16000
        audio = np.random.randn(sample_rate)  # 1 second of noise
        
        features = processor.extract_features(audio, sample_rate)
        print(f"✅ Feature extraction works ({len(features)} features)")
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
    
    try:
        # Test Whisper
        from src.speech_to_text import WhisperTranscriber
        transcriber = WhisperTranscriber(model_size="tiny")  # Use tiny for testing
        print("✅ Whisper transcriber loaded")
        
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
    
    try:
        # Test emotion recognition
        from src.emotion_recognition import EmotionRecognizer
        recognizer = EmotionRecognizer()
        print("✅ Emotion recognizer loaded")
        
    except Exception as e:
        print(f"❌ Emotion recognition test failed: {e}")

def main():
    """Run all Windows-specific fixes and tests"""
    print("🔧 Windows Compatibility Fixes")
    print("=" * 40)
    
    # Check system requirements
    print("\n1. Checking system requirements...")
    ffmpeg_ok = check_ffmpeg()
    temp_ok = fix_temp_directory()
    ollama_ok = check_ollama_models()
    
    # Create test audio
    print("\n2. Creating test audio...")
    test_audio = create_test_audio()
    
    # Test components
    print("\n3. Testing components...")
    test_pipeline_components()
    
    # Summary
    print("\n" + "=" * 40)
    print("🎯 Windows Compatibility Summary:")
    print(f"  FFmpeg: {'✅' if ffmpeg_ok else '❌'}")
    print(f"  Temp Directory: {'✅' if temp_ok else '❌'}")
    print(f"  Ollama: {'✅' if ollama_ok else '❌'}")
    print(f"  Test Audio: {'✅' if test_audio else '❌'}")
    
    if all([ffmpeg_ok, temp_ok]):
        print("\n🎉 Windows setup looks good!")
        print("🚀 Try running: python run.py")
    else:
        print("\n⚠️  Some issues need to be fixed first.")
        print("📖 Check the messages above for specific instructions.")

if __name__ == "__main__":
    main()