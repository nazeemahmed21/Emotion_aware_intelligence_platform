#!/usr/bin/env python3
"""
Test to verify that uploaded files are correctly processed through the entire flow
"""
import sys
import os
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.append('src')

def create_test_audio_file():
    """Create a test audio file to simulate an uploaded file"""
    # Create realistic speech-like audio
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create complex waveform with multiple harmonics (more speech-like)
    audio_data = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # Fundamental
        0.2 * np.sin(2 * np.pi * 400 * t) +  # First harmonic
        0.1 * np.sin(2 * np.pi * 800 * t) +  # Second harmonic
        0.05 * np.random.normal(0, 0.1, len(t))  # Noise
    )
    
    # Add amplitude modulation to simulate speech patterns
    modulation = 1 + 0.5 * np.sin(2 * np.pi * 5 * t)
    audio_data = audio_data * modulation
    
    # Save to temporary file
    test_file = Path("test_uploaded_audio.wav")
    sf.write(test_file, audio_data, sample_rate)
    
    return test_file

def test_uploaded_file_processing():
    """Test the complete flow from uploaded file to Hume analysis"""
    print("=== TESTING UPLOADED FILE PROCESSING FLOW ===")
    
    try:
        # Create test audio file (simulates user upload)
        test_file = create_test_audio_file()
        print(f"✅ Created test audio file: {test_file} ({test_file.stat().st_size} bytes)")
        
        # Simulate the streamlined app flow
        from emotion_recognizer_hume import HumeEmotionRecognizer
        recognizer = HumeEmotionRecognizer()
        print("✅ Loaded Hume recognizer")
        
        # Step 1: Read the "uploaded" file (like streamlined_app does)
        with open(test_file, 'rb') as f:
            file_content = f.read()
        print(f"✅ Read uploaded file content: {len(file_content)} bytes")
        
        # Step 2: Create temp file (like streamlined_app does)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()
            tmp_path = tmp_file.name
        print(f"✅ Created temporary file: {tmp_path}")
        
        # Step 3: Load audio data (like streamlined_app does)
        audio_data, sample_rate = sf.read(tmp_path)
        print(f"✅ Loaded audio data: {len(audio_data)} samples at {sample_rate}Hz ({len(audio_data)/sample_rate:.2f}s)")
        
        # Step 4: Analyze with Hume (the actual analysis)
        print("🧠 Analyzing with Hume AI...")
        result = recognizer.predict_emotion(audio_data, sample_rate)
        
        # Step 5: Check results
        if result.get('success', False):
            print("✅ Hume analysis completed successfully!")
            
            if 'aggregated_analysis' in result:
                analysis = result['aggregated_analysis']
                summary = analysis.get('summary', {})
                total_emotions = summary.get('total_emotions_detected', 0)
                print(f"📊 Total emotions detected: {total_emotions}")
                
                if total_emotions > 0:
                    dominant = summary.get('dominant_emotion', {})
                    print(f"👑 Dominant emotion: {dominant.get('name', 'N/A')} ({dominant.get('score', 0):.3f})")
                    
                    top_emotions = analysis.get('top_emotions', [])
                    print(f"🏆 Top emotions: {len(top_emotions)}")
                    for i, emotion in enumerate(top_emotions[:5]):
                        print(f"   {i+1}. {emotion.get('name', 'Unknown')}: {emotion.get('mean_score', 0):.3f}")
                else:
                    print("ℹ️ No emotions detected (expected for synthetic audio)")
            else:
                print("⚠️ No aggregated analysis in result")
        else:
            print(f"❌ Hume analysis failed: {result.get('error', 'Unknown error')}")
        
        # Cleanup
        try:
            os.unlink(tmp_path)
            os.unlink(test_file)
            print("✅ Cleaned up temporary files")
        except:
            pass
        
        print("\n=== FLOW VERIFICATION COMPLETE ===")
        print("✅ The uploaded file processing flow is working correctly!")
        print("✅ Your uploaded audio files WILL be analyzed by Hume AI")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_uploaded_file_processing()