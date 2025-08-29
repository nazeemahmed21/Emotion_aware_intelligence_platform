"""
Emotion recognition from audio using Hugging Face transformers
"""
import numpy as np
import torch
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import librosa
import logging
from typing import Dict, List, Tuple, Union, Optional
import tempfile
import warnings

from config import EMOTION_CONFIG
from src.audio_utils import AudioProcessor

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class EmotionRecognizer:
    """Emotion recognition using pre-trained Hugging Face models"""
    
    def __init__(self, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """
        Initialize emotion recognizer
        
        Args:
            model_name: Hugging Face model name for emotion recognition
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.pipeline = None
        self.audio_processor = AudioProcessor()
        self.emotion_mapping = {
            'LABEL_0': 'angry',
            'LABEL_1': 'calm', 
            'LABEL_2': 'disgust',
            'LABEL_3': 'fearful',
            'LABEL_4': 'happy',
            'LABEL_5': 'neutral',
            'LABEL_6': 'sad',
            'LABEL_7': 'surprised'
        }
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the emotion recognition model"""
        try:
            logger.info(f"Loading emotion recognition model: {self.model_name}")
            
            # Try to load with pipeline first (simpler)
            try:
                self.pipeline = pipeline(
                    "audio-classification",
                    model=self.model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Model loaded successfully with pipeline")
                return
            except Exception as e:
                logger.warning(f"Pipeline loading failed, trying manual loading: {e}")
            
            # Manual loading as fallback
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model moved to GPU")
            
            logger.info("Model loaded successfully with manual loading")
            
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            logger.info("Falling back to rule-based emotion detection")
            self.pipeline = None
            self.model = None
            self.processor = None
    
    def predict_emotion(self, audio_input: Union[str, bytes, np.ndarray]) -> Dict[str, Union[str, float, List]]:
        """
        Predict emotion from audio
        
        Args:
            audio_input: Audio file path, bytes, or numpy array
            
        Returns:
            Dictionary with emotion prediction results
        """
        try:
            # Load and preprocess audio
            y, sr = self.audio_processor.load_audio(audio_input)
            y = self.audio_processor.preprocess_audio(y, sr)
            
            # Resample to 16kHz if needed (required by most models)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Use pipeline if available
            if self.pipeline is not None:
                return self._predict_with_pipeline(y, sr)
            
            # Use manual model if available
            elif self.model is not None and self.processor is not None:
                return self._predict_with_model(y, sr)
            
            # Fallback to rule-based detection
            else:
                return self._predict_rule_based(y, sr)
                
        except Exception as e:
            logger.error(f"Error in emotion prediction: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.5,
                'all_emotions': [{'emotion': 'neutral', 'confidence': 0.5}],
                'error': str(e)
            }
    
    def _predict_with_pipeline(self, y: np.ndarray, sr: int) -> Dict:
        """Predict using Hugging Face pipeline"""
        try:
            # Save audio to temporary file for pipeline
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                import soundfile as sf
                sf.write(tmp_file.name, y, sr)
                
                # Get predictions
                predictions = self.pipeline(tmp_file.name)
                
                # Clean up
                import os
                os.unlink(tmp_file.name)
            
            # Process results
            if predictions:
                # Sort by confidence
                predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
                
                # Map labels to emotion names
                processed_predictions = []
                for pred in predictions:
                    emotion = self.emotion_mapping.get(pred['label'], pred['label'].lower())
                    processed_predictions.append({
                        'emotion': emotion,
                        'confidence': float(pred['score'])
                    })
                
                return {
                    'emotion': processed_predictions[0]['emotion'],
                    'confidence': processed_predictions[0]['confidence'],
                    'all_emotions': processed_predictions
                }
            
        except Exception as e:
            logger.error(f"Pipeline prediction error: {e}")
            
        return self._predict_rule_based(y, sr)
    
    def _predict_with_model(self, y: np.ndarray, sr: int) -> Dict:
        """Predict using manual model loading"""
        try:
            # Prepare input
            inputs = self.processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            predictions = predictions.cpu().numpy()[0]
            
            # Create emotion predictions
            emotion_scores = []
            for i, score in enumerate(predictions):
                emotion = self.emotion_mapping.get(f'LABEL_{i}', f'emotion_{i}')
                emotion_scores.append({
                    'emotion': emotion,
                    'confidence': float(score)
                })
            
            # Sort by confidence
            emotion_scores = sorted(emotion_scores, key=lambda x: x['confidence'], reverse=True)
            
            return {
                'emotion': emotion_scores[0]['emotion'],
                'confidence': emotion_scores[0]['confidence'],
                'all_emotions': emotion_scores
            }
            
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            
        return self._predict_rule_based(y, sr)
    
    def _predict_rule_based(self, y: np.ndarray, sr: int) -> Dict:
        """
        Rule-based emotion detection using audio features
        Fallback when ML models are not available
        """
        try:
            logger.info("Using rule-based emotion detection")
            
            # Extract features
            features = self.audio_processor.extract_features(y, sr)
            
            # Rule-based emotion detection
            emotion_scores = {}
            
            # Energy-based rules
            rms_energy = features.get('rms_energy', 0.0)
            zcr = features.get('zero_crossing_rate', 0.0)
            spectral_centroid = features.get('spectral_centroid', 0.0)
            pitch_mean = features.get('pitch_mean', 0.0)
            pitch_std = features.get('pitch_std', 0.0)
            
            # Normalize features for rule-based detection
            rms_norm = min(1.0, rms_energy * 10)  # Rough normalization
            zcr_norm = min(1.0, zcr * 100)
            
            # Happy: High energy, high pitch variation, moderate ZCR
            emotion_scores['happy'] = (
                rms_norm * 0.4 + 
                min(1.0, pitch_std / 50) * 0.3 + 
                (1.0 - abs(zcr_norm - 0.5)) * 0.3
            )
            
            # Sad: Low energy, low pitch, low variation
            emotion_scores['sad'] = (
                (1.0 - rms_norm) * 0.4 + 
                (1.0 - min(1.0, pitch_mean / 200)) * 0.3 +
                (1.0 - min(1.0, pitch_std / 30)) * 0.3
            )
            
            # Angry: High energy, high ZCR, high spectral centroid
            emotion_scores['angry'] = (
                rms_norm * 0.4 + 
                zcr_norm * 0.3 + 
                min(1.0, spectral_centroid / 3000) * 0.3
            )
            
            # Neutral: Moderate values across all features
            emotion_scores['neutral'] = (
                (1.0 - abs(rms_norm - 0.5)) * 0.4 +
                (1.0 - abs(zcr_norm - 0.3)) * 0.3 +
                (1.0 - abs(min(1.0, pitch_std / 40) - 0.5)) * 0.3
            )
            
            # Calm: Low energy, stable pitch, low ZCR
            emotion_scores['calm'] = (
                (1.0 - rms_norm) * 0.4 + 
                (1.0 - min(1.0, pitch_std / 25)) * 0.4 +
                (1.0 - zcr_norm) * 0.2
            )
            
            # Excited: Very high energy, high pitch variation
            emotion_scores['excited'] = (
                min(1.0, rms_norm * 1.2) * 0.5 + 
                min(1.0, pitch_std / 60) * 0.5
            )
            
            # Fearful: Moderate energy, high pitch, high variation
            emotion_scores['fearful'] = (
                rms_norm * 0.3 + 
                min(1.0, pitch_mean / 250) * 0.4 +
                min(1.0, pitch_std / 45) * 0.3
            )
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            
            # Sort by confidence
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Format results
            all_emotions = [{'emotion': emotion, 'confidence': score} for emotion, score in sorted_emotions]
            
            return {
                'emotion': sorted_emotions[0][0],
                'confidence': sorted_emotions[0][1],
                'all_emotions': all_emotions,
                'method': 'rule_based'
            }
            
        except Exception as e:
            logger.error(f"Rule-based prediction error: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.5,
                'all_emotions': [{'emotion': 'neutral', 'confidence': 0.5}],
                'error': str(e)
            }
    
    def batch_predict(self, audio_files: List[Union[str, bytes, np.ndarray]]) -> List[Dict]:
        """
        Predict emotions for multiple audio files
        
        Args:
            audio_files: List of audio inputs
            
        Returns:
            List of emotion predictions
        """
        results = []
        for i, audio_file in enumerate(audio_files):
            logger.info(f"Processing audio {i+1}/{len(audio_files)}")
            result = self.predict_emotion(audio_file)
            results.append(result)
        return results
    
    def get_emotion_confidence_threshold(self, emotion: str) -> float:
        """
        Get confidence threshold for specific emotion
        
        Args:
            emotion: Emotion name
            
        Returns:
            Confidence threshold
        """
        thresholds = {
            'happy': 0.6,
            'sad': 0.7,
            'angry': 0.8,
            'neutral': 0.5,
            'calm': 0.6,
            'excited': 0.7,
            'fearful': 0.8
        }
        return thresholds.get(emotion, EMOTION_CONFIG["confidence_threshold"])
    
    def is_emotion_confident(self, emotion: str, confidence: float) -> bool:
        """
        Check if emotion prediction is confident enough
        
        Args:
            emotion: Predicted emotion
            confidence: Confidence score
            
        Returns:
            True if confident enough
        """
        threshold = self.get_emotion_confidence_threshold(emotion)
        return confidence >= threshold

# Convenience functions
def predict_emotion_from_file(file_path: str) -> Dict:
    """
    Quick emotion prediction from audio file
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Emotion prediction result
    """
    recognizer = EmotionRecognizer()
    return recognizer.predict_emotion(file_path)

# Global recognizer instance
_global_recognizer = None

def get_emotion_recognizer() -> EmotionRecognizer:
    """
    Get global emotion recognizer instance (singleton pattern)
    
    Returns:
        EmotionRecognizer instance
    """
    global _global_recognizer
    if _global_recognizer is None:
        _global_recognizer = EmotionRecognizer()
    return _global_recognizer