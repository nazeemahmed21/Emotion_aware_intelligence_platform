#!/usr/bin/env python3
"""
Create test audio files for debugging the emotion-aware voice bot
"""
import numpy as np
import soundfile as sf
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

def create_test_audio_files():
    """Create various test audio files"""
    
    # Create data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    sample_rate = 16000
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Test 1: Simple sine wave (neutral)
    audio1 = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
    sf.write(os.path.join(data_dir, 'test_neutral.wav'), audio1, sample_rate)
    print("✅ Created test_neutral.wav")
    
    # Test 2: Higher frequency (happy/excited)
    audio2 = 0.3 * np.sin(2 * np.pi * 800 * t) + 0.1 * np.sin(2 * np.pi * 1200 * t)
    sf.write(os.path.join(data_dir, 'test_happy.wav'), audio2, sample_rate)
    print("✅ Created test_happy.wav")
    
    # Test 3: Lower frequency with noise (sad)
    audio3 = 0.2 * np.sin(2 * np.pi * 200 * t) + 0.05 * np.random.randn(len(t))
    sf.write(os.path.join(data_dir, 'test_sad.wav'), audio3, sample_rate)
    print("✅ Created test_sad.wav")
    
    # Test 4: Complex waveform (angry)
    audio4 = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t) + 0.1 * np.random.randn(len(t))
    # Add some "roughness" to simulate angry speech
    for i in range(0, len(audio4), sample_rate // 10):  # Every 0.1 seconds
        if i + sample_rate // 20 < len(audio4):
            audio4[i:i + sample_rate // 20] *= 1.5  # Amplitude spikes
    audio4 = np.clip(audio4, -1, 1)  # Prevent clipping
    sf.write(os.path.join(data_dir, 'test_angry.wav'), audio4, sample_rate)
    print("✅ Created test_angry.wav")
    
    # Test 5: Very quiet (calm)
    audio5 = 0.1 * np.sin(2 * np.pi * 220 * t) * np.exp(-t * 0.5)  # Decaying sine
    sf.write(os.path.join(data_dir, 'test_calm.wav'), audio5, sample_rate)
    print("✅ Created test_calm.wav")
    
    print(f"\n🎵 Created 5 test audio files in '{data_dir}' directory")
    print("You can use these files to test the emotion recognition system")

def test_audio_loading():
    """Test loading the created audio files"""
    print("\n🧪 Testing audio loading...")
    
    try:
        # Try to import audio processor
        try:
            from audio_utils import AudioProcessor
            processor = AudioProcessor()
        except ImportError:
            print("⚠️  AudioProcessor not available, using basic soundfile loading")
            processor = None
        
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        test_files = [
            'test_neutral.wav',
            'test_happy.wav', 
            'test_sad.wav',
            'test_angry.wav',
            'test_calm.wav'
        ]
        
        for filename in test_files:
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                try:
                    if processor:
                        y, sr = processor.load_audio(file_path)
                        info = processor.get_audio_info(y, sr)
                        print(f"✅ {filename}: {info['duration']:.2f}s, {info['rms_level']:.3f} RMS")
                    else:
                        y, sr = sf.read(file_path)
                        duration = len(y) / sr
                        rms = np.sqrt(np.mean(y**2))
                        print(f"✅ {filename}: {duration:.2f}s, {rms:.3f} RMS")
                except Exception as e:
                    print(f"❌ {filename}: {e}")
            else:
                print(f"⚠️  {filename}: File not found")
                
    except Exception as e:
        print(f"❌ Cannot test audio loading: {e}")

def main():
    """Create test files and test loading"""
    print("🎵 Creating Test Audio Files")
    print("=" * 30)
    
    create_test_audio_files()
    test_audio_loading()
    
    print("\n💡 Usage:")
    print("1. Upload any of the test files in the Streamlit app")
    print("2. Or use them in your own tests:")
    print("   from src.pipeline import EmotionAwareVoicePipeline")
    print("   pipeline = EmotionAwareVoicePipeline()")
    print("   result = pipeline.process_audio('data/test_happy.wav')")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)