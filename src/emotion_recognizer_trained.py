#!/usr/bin/env python3
"""
Enhanced Emotion Recognizer using trained RAVDESS model
"""
import os
import numpy as np
import torch
import librosa
import json
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import logging

logger = logging.getLogger(__name__)

class TrainedEmotionRecognizer:
    """Emotion recognizer using custom trained model"""
    
    def __init__(self, model_path="models/emotion_ravdess_actor01", fallback_model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        self.model_path = model_path
        self.fallback_model = fallback_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.feature_extractor = None
        self.label_mapping = None
        self.is_trained_model = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model or fallback to pretrained"""
        try:
            # Try to load trained model first
            if os.path.exists(self.model_path) and os.path.exists(f"{self.model_path}/training_info.json"):
                logger.info(f"Loading trained model from {self.model_path}")
                
                # Load training info
                with open(f"{self.model_path}/training_info.json", 'r') as f:
                    training_info = json.load(f)
                
                self.label_mapping = training_info['label_mapping']
                
                # Load model and feature extractor
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_path)
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
                
                self.model.to(self.device)
                self.model.eval()
                
                self.is_trained_model = True
                logger.info("✅ Trained model loaded successfully")
                logger.info(f"Emotions: {list(self.label_mapping['id2label'].values())}")
                
            else:
                raise FileNotFoundError("Trained model not found")
                
        except Exception as e:
            logger.warning(f"Could not load trained model: {e}")
            logger.info(f"Falling back to pretrained model: {self.fallback_model}")
            
            # Load fallback model
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.fallback_model)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.fallback_model)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Default emotion mapping for fallback
            self.label_mapping = {
                'id2label': {
                    0: 'angry',
                    1: 'calm', 
                    2: 'disgust',
                    3: 'fearful',
                    4: 'happy',
                    5: 'neutral',
                    6: 'sad',
                    7: 'surprised'
                }
            }
            
            self.is_trained_model = False
            logger.info("✅ Fallback model loaded")
    
    def preprocess_audio(self, audio_data, sample_rate=16000):
        """Preprocess audio for emotion recognition"""
        try:
            # Convert to numpy if needed
            if isinstance(audio_data, (list, tuple)):
                audio_data = np.array(audio_data)
            
            # Ensure float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Resample if needed
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Normalize
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Ensure minimum length (1 second)
            min_length = 16000
            if len(audio_data) < min_length:
                audio_data = np.pad(audio_data, (0, min_length - len(audio_data)))
            
            # Limit maximum length (10 seconds)
            max_length = 16000 * 10
            if len(audio_data) > max_length:
                audio_data = audio_data[:max_length]
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return None
    
    def predict_emotion(self, audio_data, sample_rate=16000):
        """Predict emotion from audio data"""
        try:
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_data, sample_rate)
            if processed_audio is None:
                return self._create_error_result("Audio preprocessing failed")
            
            # Extract features
            inputs = self.feature_extractor(
                processed_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get results
            predictions_np = predictions.cpu().numpy()[0]
            predicted_class = np.argmax(predictions_np)
            confidence = float(predictions_np[predicted_class])
            
            # Map to emotion label
            emotion = self.label_mapping['id2label'].get(str(predicted_class), 'unknown')
            
            # Get all emotion scores
            all_emotions = []
            for i, score in enumerate(predictions_np):
                emotion_name = self.label_mapping['id2label'].get(str(i), f'emotion_{i}')
                all_emotions.append({
                    'emotion': emotion_name,
                    'confidence': float(score)
                })
            
            # Sort by confidence
            all_emotions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'all_emotions': all_emotions,
                'model_type': 'trained' if self.is_trained_model else 'pretrained',
                'audio_duration': len(processed_audio) / 16000
            }
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error_message):
        """Create error result"""
        return {
            'success': False,
            'error': error_message,
            'emotion': 'neutral',
            'confidence': 0.0,
            'all_emotions': [{'emotion': 'neutral', 'confidence': 0.0}],
            'model_type': 'trained' if self.is_trained_model else 'pretrained'
        }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path if self.is_trained_model else self.fallback_model,
            'is_trained': self.is_trained_model,
            'emotions': list(self.label_mapping['id2label'].values()),
            'num_emotions': len(self.label_mapping['id2label']),
            'device': str(self.device)
        }

def test_trained_model():
    """Test the trained emotion recognizer"""
    print("🧪 Testing Trained Emotion Recognizer")
    print("=" * 45)
    
    # Initialize recognizer
    recognizer = TrainedEmotionRecognizer()
    
    # Show model info
    info = recognizer.get_model_info()
    print(f"Model Type: {'Trained' if info['is_trained'] else 'Pretrained'}")
    print(f"Emotions: {info['emotions']}")
    print(f"Device: {info['device']}")
    
    # Test with RAVDESS files if available
    test_dir = "data/Actor_01"
    if os.path.exists(test_dir):
        print(f"\n🎵 Testing with RAVDESS files from {test_dir}")
        
        # Test a few files from each emotion
        emotion_map = {
            "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
        }
        
        tested_emotions = set()
        for filename in os.listdir(test_dir)[:10]:  # Test first 10 files
            if filename.endswith('.wav'):
                file_path = os.path.join(test_dir, filename)
                
                # Parse expected emotion
                parts = filename.split('-')
                expected_emotion = emotion_map.get(parts[2], 'unknown') if len(parts) > 2 else 'unknown'
                
                if expected_emotion not in tested_emotions:
                    tested_emotions.add(expected_emotion)
                    
                    try:
                        # Load audio
                        audio, sr = librosa.load(file_path, sr=16000)
                        
                        # Predict emotion
                        result = recognizer.predict_emotion(audio, sr)
                        
                        if result['success']:
                            predicted = result['emotion']
                            confidence = result['confidence']
                            correct = "✅" if predicted == expected_emotion else "❌"
                            
                            print(f"{correct} {filename}")
                            print(f"   Expected: {expected_emotion}, Predicted: {predicted} ({confidence:.3f})")
                            
                            # Show top 3 predictions
                            print("   Top 3 predictions:")
                            for i, pred in enumerate(result['all_emotions'][:3]):
                                print(f"     {i+1}. {pred['emotion']}: {pred['confidence']:.3f}")
                        else:
                            print(f"❌ {filename}: {result['error']}")
                            
                    except Exception as e:
                        print(f"❌ Error testing {filename}: {e}")
                    
                    print()
    
    else:
        print(f"⚠️  Test directory {test_dir} not found")
        
        # Test with synthetic audio
        print("\n🎵 Testing with synthetic audio")
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        test_cases = [
            ("Happy (high freq)", 0.3 * np.sin(2 * np.pi * 800 * t)),
            ("Sad (low freq)", 0.2 * np.sin(2 * np.pi * 200 * t) + 0.05 * np.random.randn(len(t))),
            ("Neutral (mid freq)", 0.3 * np.sin(2 * np.pi * 440 * t))
        ]
        
        for test_name, audio in test_cases:
            result = recognizer.predict_emotion(audio, sample_rate)
            if result['success']:
                print(f"🎵 {test_name}")
                print(f"   Predicted: {result['emotion']} ({result['confidence']:.3f})")
            else:
                print(f"❌ {test_name}: {result['error']}")

if __name__ == "__main__":
    test_trained_model()