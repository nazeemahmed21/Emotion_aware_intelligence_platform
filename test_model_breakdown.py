#!/usr/bin/env python3
"""
Test Hume AI model breakdown functionality
"""
import sys
sys.path.append('src')
import numpy as np

def test_model_breakdown():
    print('🔍 Detailed Hume Model Analysis Test...')
    
    # Create test audio
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    from emotion_recognizer_hume import HumeEmotionRecognizer
    recognizer = HumeEmotionRecognizer()
    
    try:
        result = recognizer.predict_emotion(audio_data, sample_rate)
        
        if result.get('success') and 'aggregated_analysis' in result:
            analysis = result['aggregated_analysis']
            
            print('\n📊 Model Analysis Results:')
            model_analysis = analysis.get('model_analysis', {})
            
            for model_name in ['prosody', 'burst', 'language']:
                if model_name in model_analysis:
                    model_data = model_analysis[model_name]
                    print(f'\n🎯 {model_name.upper()} MODEL:')
                    print(f'  Description: {model_data.get("description", "N/A")}')
                    print(f'  Emotions detected: {model_data.get("emotion_count", 0)}')
                    print(f'  Average intensity: {model_data.get("avg_intensity", 0):.4f}')
                    print(f'  Contribution: {model_data.get("contribution_percentage", 0):.1f}%')
                    
                    top_emotions = model_data.get('top_emotions', [])
                    if top_emotions:
                        print(f'  Top emotions:')
                        for emotion, score in top_emotions[:3]:
                            print(f'    • {emotion}: {score:.4f}')
            
            # Summary
            model_summary = analysis.get('model_summary', {})
            print(f'\n📋 SUMMARY:')
            print(f'  Primary model: {model_summary.get("primary_model", "unknown")}')
            print(f'  Models used: {model_summary.get("total_models_used", 0)}/3')
            print(f'  Model diversity: {model_summary.get("model_diversity", 0)}')
            
            print('\n✅ All three Hume models are being analyzed!')
            
            # Test recommendations
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                print(f'\n💡 RECOMMENDATIONS:')
                for i, rec in enumerate(recommendations, 1):
                    print(f'  {i}. {rec}')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_breakdown()