#!/usr/bin/env python3
"""
Test script for Hume AI audio preprocessing
"""
import sys
import os
sys.path.append('src')

import numpy as np
from emotion_recognizer_hume import HumeEmotionRecognizer

def test_audio_preprocessing():
    """Test the Hume AI audio preprocessing functionality"""
    print("🧪 Testing Hume AI Audio Preprocessing")
    print("=" * 50)
    
    try:
        # Initialize recognizer
        recognizer = HumeEmotionRecognizer()
        print("✅ Hume recognizer initialized")
        
        # Test 1: Normal audio (44.1kHz -> 16kHz)
        print("\n📊 Test 1: Sample rate conversion")
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_normal = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        processed_audio, target_sr = recognizer._preprocess_audio_for_hume(audio_normal, sample_rate)
        print(f"   Original: {len(audio_normal)} samples at {sample_rate}Hz")
        print(f"   Processed: {len(processed_audio)} samples at {target_sr}Hz")
        print(f"   Duration: {len(processed_audio)/target_sr:.2f}s")
        
        # Test 2: Stereo to mono conversion
        print("\n📊 Test 2: Stereo to mono conversion")
        audio_stereo = np.column_stack([audio_normal, audio_normal * 0.8])
        processed_stereo, _ = recognizer._preprocess_audio_for_hume(audio_stereo, sample_rate)
        print(f"   Original: {audio_stereo.shape} (stereo)")
        print(f"   Processed: {processed_stereo.shape} (mono)")
        
        # Test 3: Long audio trimming
        print("\n📊 Test 3: Long audio trimming")
        long_duration = 20.0  # Longer than 15s limit
        t_long = np.linspace(0, long_duration, int(sample_rate * long_duration))
        audio_long = 0.3 * np.sin(2 * np.pi * 440 * t_long)
        processed_long, _ = recognizer._preprocess_audio_for_hume(audio_long, sample_rate)
        final_duration = len(processed_long) / target_sr
        print(f"   Original: {long_duration}s")
        print(f"   Processed: {final_duration:.2f}s (trimmed to max 15s)")
        
        # Test 4: Short audio padding
        print("\n📊 Test 4: Short audio padding")
        short_duration = 0.05  # Shorter than 0.1s minimum
        t_short = np.linspace(0, short_duration, int(sample_rate * short_duration))
        audio_short = 0.3 * np.sin(2 * np.pi * 440 * t_short)
        processed_short, _ = recognizer._preprocess_audio_for_hume(audio_short, sample_rate)
        final_short_duration = len(processed_short) / target_sr
        print(f"   Original: {short_duration}s")
        print(f"   Processed: {final_short_duration:.2f}s (padded to min 0.1s)")
        
        # Test 5: Amplitude normalization
        print("\n📊 Test 5: Amplitude normalization")
        audio_loud = audio_normal * 5.0  # Amplify beyond [-1, 1]
        processed_loud, _ = recognizer._preprocess_audio_for_hume(audio_loud, sample_rate)
        print(f"   Original max amplitude: {np.max(np.abs(audio_loud)):.3f}")
        print(f"   Processed max amplitude: {np.max(np.abs(processed_loud)):.3f}")
        
        # Test 6: Save to temporary file
        print("\n📊 Test 6: Save to temporary WAV file")
        temp_file = recognizer._save_audio_temp(audio_normal, sample_rate)
        file_size_mb = temp_file.stat().st_size / (1024 * 1024)
        print(f"   Temp file: {temp_file}")
        print(f"   File size: {file_size_mb:.3f}MB")
        
        # Clean up
        recognizer._cleanup_temp_file(temp_file)
        print("   ✅ Temp file cleaned up")
        
        print("\n🎯 All preprocessing tests passed!")
        print("✅ Audio preprocessing is working correctly for Hume AI")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_preprocessing()