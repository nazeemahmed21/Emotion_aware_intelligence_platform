#!/usr/bin/env python3
"""
Backend Integration Example for React Frontend
This shows how to create an API endpoint that works with the React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import sys
from pathlib import Path

# Add your existing code path
sys.path.append(str(Path(__file__).parent))

# Import your existing Hume analysis functions
# from emotion_aware_voice_analyzer import analyze_with_hume, extract_emotions

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

@app.route('/api/analyze', methods=['POST'])
def analyze_emotion():
    """
    API endpoint for emotion analysis
    Receives audio file from React frontend and returns emotion analysis
    """
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # TODO: Replace this with your actual Hume analysis
            # predictions = analyze_with_hume(tmp_path)
            # emotions = extract_emotions(predictions)
            
            # For now, return mock data that matches your existing structure
            emotions = get_mock_emotions()
            
            # Transform to frontend format
            response_data = transform_for_frontend(emotions)
            
            return jsonify(response_data)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except (PermissionError, FileNotFoundError):
                pass
    
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return jsonify({"error": str(e)}), 500

def get_mock_emotions():
    """
    Mock emotion data that matches your existing structure
    Replace this with your actual extract_emotions() function
    """
    return [
        {'name': 'joy', 'score': 0.85, 'model': 'prosody'},
        {'name': 'excitement', 'score': 0.72, 'model': 'burst'},
        {'name': 'confidence', 'score': 0.68, 'model': 'language'},
        {'name': 'curiosity', 'score': 0.45, 'model': 'prosody'},
        {'name': 'calmness', 'score': 0.38, 'model': 'language'},
        {'name': 'surprise', 'score': 0.32, 'model': 'burst'},
        {'name': 'interest', 'score': 0.28, 'model': 'prosody'},
        {'name': 'contentment', 'score': 0.25, 'model': 'language'},
        {'name': 'anticipation', 'score': 0.22, 'model': 'burst'},
        {'name': 'satisfaction', 'score': 0.18, 'model': 'prosody'}
    ]

def transform_for_frontend(emotions):
    """
    Transform emotion data to match frontend expectations
    """
    if not emotions:
        return {
            "emotions": [],
            "summary": None,
            "aiResponse": "No emotions detected in the audio."
        }
    
    # Calculate summary statistics
    emotion_stats = {}
    for emotion in emotions:
        name = emotion['name']
        score = emotion['score']
        model = emotion['model']
        
        if name not in emotion_stats:
            emotion_stats[name] = {
                'scores': [],
                'models': [],
                'total_score': 0,
                'count': 0
            }
        
        emotion_stats[name]['scores'].append(score)
        emotion_stats[name]['models'].append(model)
        emotion_stats[name]['total_score'] += score
        emotion_stats[name]['count'] += 1
    
    # Calculate final stats
    for name, stats in emotion_stats.items():
        stats['mean_score'] = stats['total_score'] / stats['count']
        stats['max_score'] = max(stats['scores'])
        stats['unique_models'] = list(set(stats['models']))
    
    # Find dominant emotion
    dominant = max(emotion_stats.items(), key=lambda x: x[1]['mean_score'])
    
    # Calculate model breakdown
    model_breakdown = {'prosody': {'emotionCount': 0, 'totalScore': 0, 'avgScore': 0},
                      'burst': {'emotionCount': 0, 'totalScore': 0, 'avgScore': 0},
                      'language': {'emotionCount': 0, 'totalScore': 0, 'avgScore': 0}}
    
    for emotion in emotions:
        model = emotion['model']
        if model in model_breakdown:
            model_breakdown[model]['emotionCount'] += 1
            model_breakdown[model]['totalScore'] += emotion['score']
    
    # Calculate averages
    for model_data in model_breakdown.values():
        if model_data['emotionCount'] > 0:
            model_data['avgScore'] = model_data['totalScore'] / model_data['emotionCount']
    
    # Hesitancy analysis
    hesitancy_keywords = ['anxiety', 'nervousness', 'uncertainty', 'confusion', 
                         'hesitation', 'doubt', 'worry', 'stress', 'awkwardness']
    
    hesitancy_emotions = []
    for name, stats in emotion_stats.items():
        if any(keyword in name.lower() for keyword in hesitancy_keywords):
            hesitancy_emotions.append((name, stats['mean_score']))
    
    hesitancy_analysis = {
        'indicators': len(hesitancy_emotions),
        'averageScore': sum(score for _, score in hesitancy_emotions) / len(hesitancy_emotions) if hesitancy_emotions else 0,
        'level': 'High' if not hesitancy_emotions or sum(score for _, score in hesitancy_emotions) / len(hesitancy_emotions) < 0.5 else 'Low',
        'patterns': [name for name, _ in hesitancy_emotions]
    }
    
    # Generate AI response
    avg_intensity = sum(e['score'] for e in emotions) / len(emotions)
    ai_response = f"Based on my analysis of your voice, I detected {len(emotion_stats)} distinct emotions. "
    ai_response += f"The most prominent emotion was **{dominant[0]}** with a confidence of {dominant[1]['mean_score']*100:.1f}%. "
    
    if avg_intensity > 0.6:
        ai_response += "Your speech shows high emotional intensity, suggesting strong feelings or engagement with the topic. "
    elif avg_intensity > 0.3:
        ai_response += "Your emotional expression appears moderate and balanced. "
    else:
        ai_response += "Your speech shows relatively calm and controlled emotional expression. "
    
    if hesitancy_analysis['indicators'] > 0:
        ai_response += f"I also noticed {hesitancy_analysis['indicators']} indicators of hesitancy or uncertainty."
    else:
        ai_response += "Your speech appears confident and decisive without significant hesitancy patterns."
    
    return {
        "emotions": emotions,
        "summary": {
            "totalEmotions": len(emotion_stats),
            "dominantEmotion": [dominant[0], {"meanScore": dominant[1]['mean_score']}],
            "averageIntensity": avg_intensity,
            "modelBreakdown": model_breakdown,
            "hesitancyAnalysis": hesitancy_analysis
        },
        "aiResponse": ai_response
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "emotion-analysis-api"})

if __name__ == '__main__':
    print("🚀 Starting Emotion Analysis API Server...")
    print("📡 Frontend should connect to: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/api/analyze")
    print("💡 Update your React .env file: REACT_APP_API_URL=http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)