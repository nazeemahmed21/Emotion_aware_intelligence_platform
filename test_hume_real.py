#!/usr/bin/env python3
"""
Test Hume integration with more realistic audio
"""
import sys
sys.path.append('src')
from pipeline import EmotionAwareVoicePipeline
import numpy as np

def test_hume_with_realistic_audio():
    print('=== TESTING HUME WITH REALISTIC AUDIO ===')
    
    # Create more realistic test audio with speech-like characteristics
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create a more complex waveform that might trigger emotion detection
    # Mix multiple frequencies to simulate speech formants
    audio_data = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # Fundamental frequency
        0.2 * np.sin(2 * np.pi * 400 * t) +  # First harmonic
        0.1 * np.sin(2 * np.pi * 800 * t) +  # Second harmonic
        0.05 * np.random.normal(0, 0.1, len(t))  # Add some noise
    )

    # Add amplitude modulation to simulate speech patterns
    modulation = 1 + 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
    audio_data = audio_data * modulation

    pipeline = EmotionAwareVoicePipeline()
    result = pipeline.emotion_recognizer.predict_emotion(audio_data, sample_rate)

    print(f'Success: {result.get("success", False)}')
    if result.get('success'):
        if 'aggregated_analysis' in result:
            analysis = result['aggregated_analysis']
            summary = analysis.get('summary', {})
            print(f'Total emotions detected: {summary.get("total_emotions_detected", 0)}')
            
            top_emotions = analysis.get('top_emotions', [])
            print(f'Top emotions count: {len(top_emotions)}')
            
            if top_emotions:
                print('Top 5 emotions:')
                for i, emotion in enumerate(top_emotions[:5]):
                    print(f'  {i+1}. {emotion.get("name", "Unknown")}: {emotion.get("mean_score", 0):.3f}')
            
            # Check model analysis
            model_analysis = analysis.get('model_analysis', {})
            if model_analysis:
                print('Model contributions:')
                for model, data in model_analysis.items():
                    count = data.get('emotion_count', 0)
                    contrib = data.get('contribution_percentage', 0)
                    print(f'  {model}: {count} emotions ({contrib:.1f}%)')
            
            # Check if we have detailed emotions
            detailed = analysis.get('detailed_emotions', {})
            print(f'Detailed emotions available: {len(detailed)} emotions')
            
            if detailed:
                print('Sample detailed emotions:')
                for i, (name, stats) in enumerate(list(detailed.items())[:3]):
                    print(f'  {name}: mean={stats.get("mean", 0):.3f}, max={stats.get("max", 0):.3f}')
        else:
            print('No aggregated analysis found')
            print(f'Available keys: {list(result.keys())}')
    else:
        print(f'Error: {result.get("error", "Unknown error")}')

if __name__ == "__main__":
    test_hume_with_realistic_audio()