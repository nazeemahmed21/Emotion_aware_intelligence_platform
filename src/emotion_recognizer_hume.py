#!/usr/bin/env python3
"""
Hume AI Emotion Recognizer
Integrates Hume AI's emotion analysis API with the existing pipeline
"""
import os
import sys
import asyncio
import tempfile
import logging
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add knowledge_base to path for Hume imports
knowledge_base_path = Path(__file__).parent.parent / "knowledge_base"
sys.path.append(str(knowledge_base_path))

try:
    from hume.hume_client import HumeClient, HumeConfig, GranularityLevel
except ImportError as e:
    logging.error(f"Could not import Hume client: {e}")
    logging.error("Make sure the knowledge_base/hume directory is accessible")
    raise

logger = logging.getLogger(__name__)

class HumeEmotionRecognizer:
    """Emotion recognizer using Hume AI's API"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Hume emotion recognizer
        
        Args:
            config_path: Optional path to config file. If None, uses environment variables
        """
        self.config_path = config_path
        self.hume_config = None
        self.hume_client = None
        self.temp_dir = Path(tempfile.gettempdir()) / "hume_emotion_temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Emotion mapping for consistency with existing pipeline
        self.emotion_mapping = {
            # Map Hume emotions to pipeline-expected emotions
            'Joy': 'happy',
            'Happiness': 'happy',
            'Amusement': 'happy',
            'Excitement': 'happy',
            'Contentment': 'happy',
            'Satisfaction': 'happy',
            
            'Sadness': 'sad',
            'Disappointment': 'sad',
            'Distress': 'sad',
            'Pain': 'sad',
            'Empathic Pain': 'sad',
            
            'Anger': 'angry',
            'Rage': 'angry',
            'Annoyance': 'angry',
            'Contempt': 'angry',
            'Disgust': 'angry',
            'Frustration': 'angry',
            
            'Fear': 'fearful',
            'Anxiety': 'fearful',
            'Horror': 'fearful',
            'Nervousness': 'fearful',
            'Worry': 'fearful',
            
            'Surprise (positive)': 'surprised',
            'Surprise (negative)': 'surprised',
            'Surprise': 'surprised',
            'Realization': 'surprised',
            
            'Calmness': 'calm',
            'Serenity': 'calm',
            'Peacefulness': 'calm',
            'Tranquility': 'calm',
            
            'Neutral': 'neutral',
            'Boredom': 'neutral',
            'Indifference': 'neutral'
        }
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Hume client"""
        try:
            if self.config_path and Path(self.config_path).exists():
                self.hume_config = HumeConfig.from_config_file(self.config_path)
            else:
                # Use environment variables
                api_key = os.getenv('HUME_API_KEY')
                if not api_key:
                    raise ValueError("HUME_API_KEY must be set in environment variables or config file")
                
                self.hume_config = HumeConfig(
                    api_key=api_key,
                    secret_key=os.getenv('HUME_SECRET_KEY'),
                    webhook_url=os.getenv('HUME_WEBHOOK_URL')
                )
            
            self.hume_client = HumeClient(self.hume_config)
            logger.info("✅ Hume AI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hume client: {e}")
            raise
    
    def _preprocess_audio_for_hume(self, audio_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """
        Preprocess audio to meet Hume AI's exact specifications
        
        Based on Hume's requirements:
        - Format: WAV with Linear PCM encoding
        - Sample Rate: 16 kHz minimum (higher rates are resampled)
        - Bit Depth: 16-bit (handled by soundfile)
        - Channels: Mono (1 channel)
        - Duration: Recommended 3-15 seconds
        - File Size: < 10 MB
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Current sample rate
            
        Returns:
            Tuple of (preprocessed_audio, target_sample_rate)
        """
        try:
            logger.debug(f"Preprocessing audio: shape={audio_data.shape}, sr={sample_rate}")
            
            # Step 1: Convert to mono if stereo
            if len(audio_data.shape) > 1:
                if audio_data.shape[1] > 1:  # Multi-channel
                    audio_data = np.mean(audio_data, axis=1)
                    logger.debug("Converted stereo/multi-channel to mono")
                elif audio_data.shape[0] > 1 and audio_data.shape[1] == 1:  # Column vector
                    audio_data = audio_data.flatten()
            
            # Step 2: Ensure 1D array
            audio_data = np.asarray(audio_data).flatten()
            
            # Step 3: Resample to 16kHz if needed (Hume's minimum requirement)
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                try:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
                    logger.debug(f"Resampled from {sample_rate}Hz to {target_sample_rate}Hz")
                except ImportError:
                    logger.warning("Librosa not available for resampling. Using scipy instead.")
                    from scipy import signal
                    # Calculate resampling ratio
                    ratio = target_sample_rate / sample_rate
                    new_length = int(len(audio_data) * ratio)
                    audio_data = signal.resample(audio_data, new_length)
                    logger.debug(f"Resampled from {sample_rate}Hz to {target_sample_rate}Hz using scipy")
            
            # Step 4: Normalize amplitude to prevent clipping
            # Ensure values are in [-1, 1] range for 16-bit PCM
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                # Leave some headroom to prevent clipping
                audio_data = audio_data / max_val * 0.95
            
            # Step 5: Ensure proper data type (float32 for soundfile)
            audio_data = audio_data.astype(np.float32)
            
            # Step 6: Check duration and trim if too long (recommended 3-15 seconds)
            duration = len(audio_data) / target_sample_rate
            max_duration = 15.0  # seconds
            
            if duration > max_duration:
                max_samples = int(max_duration * target_sample_rate)
                audio_data = audio_data[:max_samples]
                logger.warning(f"Audio trimmed from {duration:.2f}s to {max_duration}s for optimal processing")
            
            # Step 7: Ensure minimum duration (at least 0.1 seconds)
            min_duration = 0.1
            if duration < min_duration:
                min_samples = int(min_duration * target_sample_rate)
                # Pad with silence if too short
                padding = min_samples - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant', constant_values=0)
                logger.debug(f"Audio padded from {duration:.2f}s to {min_duration}s")
            
            final_duration = len(audio_data) / target_sample_rate
            logger.debug(f"Audio preprocessing complete: duration={final_duration:.2f}s, samples={len(audio_data)}")
            
            return audio_data, target_sample_rate
            
        except Exception as e:
            logger.error(f"Error preprocessing audio for Hume: {e}")
            raise
    
    def _save_audio_temp(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Path:
        """
        Save audio data to a temporary file for Hume API with proper preprocessing
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Path to the temporary audio file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_file = self.temp_dir / f"hume_audio_{timestamp}.wav"
        
        try:
            # Preprocess audio to meet Hume's specifications
            processed_audio, target_sr = self._preprocess_audio_for_hume(audio_data, sample_rate)
            
            # Save as WAV file with Linear PCM encoding (16-bit)
            # soundfile automatically handles the Linear PCM encoding and 16-bit depth
            sf.write(
                temp_file, 
                processed_audio, 
                target_sr,
                subtype='PCM_16'  # Explicitly specify 16-bit PCM
            )
            
            # Verify file size (should be < 10MB)
            file_size_mb = temp_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 10:
                logger.warning(f"Audio file size ({file_size_mb:.2f}MB) exceeds Hume's 10MB limit")
            
            logger.debug(f"Saved Hume-compliant audio file: {temp_file} ({file_size_mb:.2f}MB)")
            return temp_file
            
        except Exception as e:
            logger.error(f"Error saving temporary audio file for Hume: {e}")
            raise
    
    def validate_audio_for_hume(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate an audio file against Hume AI's specifications
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with validation results and recommendations
        """
        try:
            audio_file = Path(audio_path)
            if not audio_file.exists():
                return {
                    'valid': False,
                    'error': f"File not found: {audio_file}",
                    'recommendations': ['Ensure the file path is correct']
                }
            
            # Load audio file info
            try:
                import librosa
                audio_data, sample_rate = librosa.load(audio_file, sr=None)
            except ImportError:
                audio_data, sample_rate = sf.read(audio_file)
            
            # Check specifications
            issues = []
            recommendations = []
            
            # Check file format
            if not audio_file.suffix.lower() == '.wav':
                issues.append(f"File format is {audio_file.suffix}, should be .wav")
                recommendations.append("Convert file to WAV format with Linear PCM encoding")
            
            # Check sample rate
            if sample_rate < 16000:
                issues.append(f"Sample rate is {sample_rate}Hz, minimum is 16kHz")
                recommendations.append("Resample audio to at least 16kHz")
            
            # Check channels
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                issues.append(f"Audio has {audio_data.shape[1]} channels, should be mono")
                recommendations.append("Convert to mono audio")
            
            # Check duration
            duration = len(audio_data) / sample_rate
            if duration < 0.1:
                issues.append(f"Duration is {duration:.2f}s, too short for reliable analysis")
                recommendations.append("Use audio clips of at least 0.1 seconds")
            elif duration > 15:
                issues.append(f"Duration is {duration:.2f}s, longer than recommended 15s")
                recommendations.append("Consider trimming to 3-15 seconds for optimal results")
            
            # Check file size
            file_size_mb = audio_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 10:
                issues.append(f"File size is {file_size_mb:.2f}MB, exceeds 10MB limit")
                recommendations.append("Compress or trim audio to under 10MB")
            
            # Check amplitude
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude > 1.0:
                issues.append(f"Audio amplitude exceeds 1.0 (max: {max_amplitude:.3f})")
                recommendations.append("Normalize audio amplitude to prevent clipping")
            elif max_amplitude < 0.01:
                issues.append(f"Audio amplitude very low (max: {max_amplitude:.3f})")
                recommendations.append("Check if audio is too quiet or contains mostly silence")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'recommendations': recommendations,
                'specs': {
                    'format': audio_file.suffix.lower(),
                    'sample_rate': sample_rate,
                    'channels': 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
                    'duration': duration,
                    'file_size_mb': file_size_mb,
                    'max_amplitude': max_amplitude,
                    'samples': len(audio_data)
                }
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Error validating audio: {e}",
                'recommendations': ['Check if the file is a valid audio file']
            }
    
    def _cleanup_temp_file(self, temp_file: Path):
        """Clean up temporary file with Windows compatibility"""
        import time
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
                    return
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    logger.debug(f"Retry {attempt + 1}: Could not delete {temp_file}, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.warning(f"Could not clean up temporary file {temp_file} after {max_retries} attempts: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error cleaning up temporary file {temp_file}: {e}")
                break
    
    def _extract_emotions_from_predictions(self, predictions: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Extract emotion data from Hume predictions with model-specific breakdown
        
        Args:
            predictions: Raw predictions from Hume API
            
        Returns:
            Tuple of (all_emotions_list, emotions_by_model_dict)
        """
        all_emotions = []
        emotions_by_model = {
            "prosody": [],    # Vocal tone and speech patterns
            "burst": [],      # Short emotional expressions
            "language": []    # Text-based emotional content
        }
        
        try:
            # Handle both list and dict response formats
            if isinstance(predictions, list):
                predictions_list = predictions
            else:
                predictions_list = predictions.get("results", {}).get("predictions", [])
            
            for file_prediction in predictions_list:
                # Handle nested structure
                if "results" in file_prediction:
                    file_predictions = file_prediction["results"].get("predictions", [])
                else:
                    file_predictions = [file_prediction]
                
                for pred in file_predictions:
                    models = pred.get("models", {})
                    
                    # Process each model separately to maintain model attribution
                    for model_name in ["prosody", "burst", "language"]:
                        if model_name not in models:
                            continue
                        
                        model_data = models[model_name]
                        grouped_predictions = model_data.get("grouped_predictions", [])
                        
                        for group in grouped_predictions:
                            predictions_list = group.get("predictions", [])
                            
                            for prediction in predictions_list:
                                if "emotions" in prediction:
                                    model_emotions = prediction["emotions"]
                                    
                                    # Add model attribution to each emotion
                                    for emotion in model_emotions:
                                        emotion_with_model = emotion.copy()
                                        emotion_with_model["model"] = model_name
                                        emotions_by_model[model_name].append(emotion_with_model)
                                        all_emotions.append(emotion_with_model)
            
            total_emotions = len(all_emotions)
            model_counts = {model: len(emotions) for model, emotions in emotions_by_model.items()}
            
            logger.debug(f"Extracted {total_emotions} total emotion predictions")
            logger.debug(f"Model breakdown: Prosody={model_counts['prosody']}, Burst={model_counts['burst']}, Language={model_counts['language']}")
            
            return all_emotions, emotions_by_model
            
        except Exception as e:
            logger.error(f"Error extracting emotions from predictions: {e}")
            return [], {"prosody": [], "burst": [], "language": []}
    
    def _aggregate_emotions(self, emotions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate emotion scores and map to pipeline emotions
        
        Args:
            emotions: List of emotion dictionaries from Hume
            
        Returns:
            Dictionary mapping pipeline emotions to aggregated scores
        """
        if not emotions:
            return {'neutral': 0.5}
        
        # Group emotions by name and calculate average scores
        emotion_scores = {}
        for emotion in emotions:
            name = emotion.get('name', '')
            score = emotion.get('score', 0.0)
            
            if name not in emotion_scores:
                emotion_scores[name] = []
            emotion_scores[name].append(score)
        
        # Calculate averages
        avg_scores = {
            name: sum(scores) / len(scores)
            for name, scores in emotion_scores.items()
        }
        
        # Map to pipeline emotions
        pipeline_emotions = {}
        for hume_emotion, score in avg_scores.items():
            pipeline_emotion = self.emotion_mapping.get(hume_emotion, 'neutral')
            
            if pipeline_emotion not in pipeline_emotions:
                pipeline_emotions[pipeline_emotion] = []
            pipeline_emotions[pipeline_emotion].append(score)
        
        # Average scores for each pipeline emotion
        final_scores = {}
        for emotion, scores in pipeline_emotions.items():
            final_scores[emotion] = sum(scores) / len(scores)
        
        # Ensure we have all expected emotions
        expected_emotions = ['happy', 'sad', 'angry', 'fearful', 'surprised', 'calm', 'neutral']
        for emotion in expected_emotions:
            if emotion not in final_scores:
                final_scores[emotion] = 0.0
        
        return final_scores
    
    def _determine_primary_emotion(self, emotion_scores: Dict[str, float]) -> tuple[str, float]:
        """
        Determine the primary emotion and its confidence
        
        Args:
            emotion_scores: Dictionary of emotion scores
            
        Returns:
            Tuple of (primary_emotion, confidence)
        """
        if not emotion_scores:
            return 'neutral', 0.5
        
        # Find the emotion with the highest score
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        return primary_emotion[0], primary_emotion[1]
    
    def _create_aggregated_analysis(self, emotions: List[Dict[str, Any]], 
                                   emotions_by_model: Dict[str, List[Dict[str, Any]]],
                                   raw_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create streamlined aggregated analysis focused on top emotions, peaks, and hesitancy
        
        Args:
            emotions: List of all emotion dictionaries from Hume
            emotions_by_model: Dictionary of emotions grouped by model
            raw_predictions: Raw predictions from Hume API
            
        Returns:
            Dictionary containing streamlined aggregated analysis
        """
        if not emotions:
            return self._create_empty_aggregated_analysis()
        
        # Group emotions by name and calculate statistics
        emotion_stats = {}
        all_scores = []
        
        for emotion in emotions:
            name = emotion.get('name', '')
            score = emotion.get('score', 0.0)
            model = emotion.get('model', 'unknown')
            all_scores.append(score)
            
            if name not in emotion_stats:
                emotion_stats[name] = {
                    'scores': [],
                    'models': [],
                    'model_breakdown': {'prosody': [], 'burst': [], 'language': []}
                }
            
            emotion_stats[name]['scores'].append(score)
            emotion_stats[name]['models'].append(model)
            emotion_stats[name]['model_breakdown'][model].append(score)
        
        # Calculate statistics for each emotion
        aggregated_emotions = {}
        for name, data in emotion_stats.items():
            scores = data['scores']
            models = data['models']
            model_breakdown = data['model_breakdown']
            
            # Calculate model-specific statistics
            model_stats = {}
            for model_name, model_scores in model_breakdown.items():
                if model_scores:
                    model_stats[model_name] = {
                        'mean': sum(model_scores) / len(model_scores),
                        'max': max(model_scores),
                        'count': len(model_scores),
                        'contribution': len(model_scores) / len(scores) * 100
                    }
                else:
                    model_stats[model_name] = {
                        'mean': 0.0,
                        'max': 0.0,
                        'count': 0,
                        'contribution': 0.0
                    }
            
            aggregated_emotions[name] = {
                'mean': sum(scores) / len(scores),
                'max': max(scores),
                'min': min(scores),
                'std': np.std(scores) if len(scores) > 1 else 0.0,
                'count': len(scores),
                'total': sum(scores),
                'models_detected': list(set(models)),
                'model_stats': model_stats
            }
        
        # Sort emotions by mean score for top 10
        sorted_emotions = sorted(aggregated_emotions.items(), 
                               key=lambda x: x[1]['mean'], reverse=True)
        
        # Get top 10 emotions
        top_emotions = sorted_emotions[:10]
        dominant_emotion = sorted_emotions[0] if sorted_emotions else ('neutral', {'mean': 0.5})
        
        # Calculate sentiment balance
        positive_emotions = ['Joy', 'Happiness', 'Amusement', 'Excitement', 'Contentment', 
                           'Satisfaction', 'Love', 'Adoration', 'Pride', 'Relief']
        negative_emotions = ['Sadness', 'Anger', 'Fear', 'Anxiety', 'Disgust', 'Contempt',
                           'Frustration', 'Disappointment', 'Distress', 'Pain']
        
        positive_score = sum(aggregated_emotions.get(emotion, {}).get('mean', 0) 
                           for emotion in positive_emotions if emotion in aggregated_emotions)
        negative_score = sum(aggregated_emotions.get(emotion, {}).get('mean', 0) 
                           for emotion in negative_emotions if emotion in aggregated_emotions)
        
        # Calculate overall metrics
        overall_intensity = np.mean(all_scores) if all_scores else 0.0
        peak_score = max(all_scores) if all_scores else 0.0
        
        # Calculate model-specific insights
        model_analysis = {}
        model_descriptions = {
            'prosody': 'Vocal tone, pitch, rhythm, and speech patterns',
            'burst': 'Short emotional expressions and vocal bursts',
            'language': 'Text-based emotional content from speech'
        }
        
        for model_name, model_emotions in emotions_by_model.items():
            if model_emotions:
                model_scores = [e.get('score', 0.0) for e in model_emotions]
                model_analysis[model_name] = {
                    'description': model_descriptions.get(model_name, ''),
                    'emotion_count': len(model_emotions),
                    'avg_intensity': np.mean(model_scores),
                    'max_intensity': np.max(model_scores),
                    'top_emotions': sorted(
                        [(e.get('name', ''), e.get('score', 0.0)) for e in model_emotions],
                        key=lambda x: x[1], reverse=True
                    )[:3],  # Top 3 emotions from this model
                    'contribution_percentage': len(model_emotions) / len(emotions) * 100 if emotions else 0
                }
            else:
                model_analysis[model_name] = {
                    'description': model_descriptions.get(model_name, ''),
                    'emotion_count': 0,
                    'avg_intensity': 0.0,
                    'max_intensity': 0.0,
                    'top_emotions': [],
                    'contribution_percentage': 0.0
                }
        
        # Create streamlined analysis
        analysis = {
            'summary': {
                'total_emotions_detected': len(aggregated_emotions),
                'dominant_emotion': {
                    'name': dominant_emotion[0],
                    'score': dominant_emotion[1]['mean'],
                    'confidence': dominant_emotion[1]['mean']
                },
                'overall_intensity': overall_intensity,
                'peak_score': peak_score,
                'positive_sentiment': positive_score,
                'negative_sentiment': negative_score,
                'sentiment_balance': positive_score - negative_score
            },
            'top_emotions': [
                {
                    'name': name,
                    'mean_score': stats['mean'],
                    'max_score': stats['max'],
                    'frequency': stats['count'],
                    'intensity_level': self._get_intensity_level(stats['mean'])
                }
                for name, stats in top_emotions
            ],
            'model_analysis': model_analysis,
            'detailed_emotions': aggregated_emotions,
            'raw_data': {
                'total_predictions': len(emotions),
                'unique_emotions': len(aggregated_emotions),
                'score_range': {
                    'min': min(all_scores) if all_scores else 0.0,
                    'max': max(all_scores) if all_scores else 0.0
                }
            }
        }
        
        return analysis
    
    def _get_intensity_level(self, score: float) -> str:
        """Determine intensity level based on score"""
        if score > 0.7:
            return 'high'
        elif score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, emotions: Dict[str, Any], 
                                positive_score: float, negative_score: float,
                                model_analysis: Dict[str, Any] = None) -> List[str]:
        """Generate recommendations based on emotional analysis"""
        recommendations = []
        
        # Check for high negative emotions
        high_negative = [name for name, stats in emotions.items() 
                        if stats['mean'] > 0.6 and name in ['Anger', 'Sadness', 'Anxiety', 'Frustration']]
        
        if high_negative:
            recommendations.append(f"High levels of {', '.join(high_negative[:2])} detected. Consider addressing underlying concerns.")
        
        # Check for positive emotions
        if positive_score > negative_score:
            recommendations.append("Overall positive emotional tone detected. Good emotional state.")
        elif negative_score > positive_score * 1.5:
            recommendations.append("Significant negative emotional content. May benefit from supportive response.")
        
        # Check for emotional variability
        if len(emotions) > 10:
            recommendations.append("High emotional complexity detected. Multiple emotions present.")
        
        # Check for specific emotion patterns
        if 'Confusion' in emotions and emotions['Confusion']['mean'] > 0.5:
            recommendations.append("Confusion detected. May need clarification or additional information.")
        
        if 'Excitement' in emotions and emotions['Excitement']['mean'] > 0.6:
            recommendations.append("High excitement detected. Positive engagement opportunity.")
        
        # Add model-specific recommendations
        if model_analysis:
            prosody_contrib = model_analysis.get('prosody', {}).get('contribution_percentage', 0)
            burst_contrib = model_analysis.get('burst', {}).get('contribution_percentage', 0)
            language_contrib = model_analysis.get('language', {}).get('contribution_percentage', 0)
            
            if prosody_contrib > 50:
                recommendations.append("Strong vocal emotion detected - tone and speech patterns are primary indicators.")
            elif burst_contrib > 40:
                recommendations.append("Emotional bursts detected - immediate reactions are prominent.")
            elif language_contrib > 40:
                recommendations.append("Text-based emotions detected - word choice conveys emotional content.")
            
            if prosody_contrib > 30 and burst_contrib > 30:
                recommendations.append("Multi-modal emotion expression - both vocal tone and emotional bursts present.")
        
        return recommendations if recommendations else ["Emotional analysis complete. No specific recommendations."]
    
    def _create_empty_aggregated_analysis(self) -> Dict[str, Any]:
        """Create empty aggregated analysis structure"""
        return {
            'summary': {
                'total_emotions_detected': 0,
                'dominant_emotion': {'name': 'neutral', 'score': 0.5, 'confidence': 0.5},
                'overall_intensity': 0.0,
                'emotional_variability': 0.0,
                'positive_sentiment': 0.0,
                'negative_sentiment': 0.0,
                'sentiment_balance': 0.0
            },
            'top_emotions': [],
            'intensity_distribution': {
                'high_intensity': {'count': 0, 'emotions': [], 'average_score': 0.0},
                'medium_intensity': {'count': 0, 'emotions': [], 'average_score': 0.0},
                'low_intensity': {'count': 0, 'emotions': [], 'average_score': 0.0}
            },
            'emotional_categories': {
                'positive': {'total_score': 0.0, 'emotions': {}},
                'negative': {'total_score': 0.0, 'emotions': {}},
                'neutral': {'score': 0.5}
            },
            'detailed_emotions': {},
            'raw_data': {
                'total_predictions': 0,
                'unique_emotions': 0,
                'score_range': {'min': 0.0, 'max': 0.0}
            },
            'recommendations': ['No emotional data available for analysis.']
        }
    
    async def predict_emotion_async(self, audio_data: Union[np.ndarray, str, Path], 
                                   sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Predict emotion from audio data using Hume AI (async version)
        
        Args:
            audio_data: Audio data as numpy array, file path, or Path object
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary containing emotion prediction results
        """
        temp_file = None
        
        try:
            # Handle different input types
            if isinstance(audio_data, (str, Path)):
                # File path provided - load and preprocess it
                audio_file = Path(audio_data)
                if not audio_file.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_file}")
                
                # Load the audio file and preprocess it
                logger.debug(f"Loading audio file for preprocessing: {audio_file}")
                try:
                    import librosa
                    audio_array, original_sr = librosa.load(audio_file, sr=None)
                except ImportError:
                    # Fallback to soundfile
                    audio_array, original_sr = sf.read(audio_file)
                
                # Preprocess and save to temp file
                temp_file = self._save_audio_temp(audio_array, original_sr)
                cleanup_temp = True
            else:
                # Numpy array provided
                temp_file = self._save_audio_temp(audio_data, sample_rate)
                cleanup_temp = True
            
            # Submit to Hume AI
            logger.debug(f"Submitting audio file to Hume AI: {temp_file}")
            job_id = await self.hume_client.submit_files(
                [str(temp_file)], 
                GranularityLevel.UTTERANCE
            )
            
            # Wait for completion
            logger.debug(f"Waiting for Hume AI job completion: {job_id}")
            success = await self.hume_client.wait_for_job(job_id)
            
            if not success:
                raise RuntimeError(f"Hume AI job failed: {job_id}")
            
            # Get predictions
            predictions = await self.hume_client.get_job_predictions(job_id, format="json")
            
            # Extract and process emotions with model breakdown
            emotions, emotions_by_model = self._extract_emotions_from_predictions(predictions)
            emotion_scores = self._aggregate_emotions(emotions)
            primary_emotion, confidence = self._determine_primary_emotion(emotion_scores)
            
            # Create comprehensive aggregated analysis with model insights
            aggregated_analysis = self._create_aggregated_analysis(emotions, emotions_by_model, predictions)
            
            # Create all_emotions list for compatibility
            all_emotions = [
                {'emotion': emotion, 'confidence': score}
                for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            
            result = {
                'success': True,
                'emotion': primary_emotion,
                'confidence': confidence,
                'all_emotions': all_emotions,
                'emotion_scores': emotion_scores,
                'aggregated_analysis': aggregated_analysis,
                'model_type': 'hume_ai',
                'job_id': job_id,
                'raw_predictions': predictions
            }
            
            logger.info(f"Hume AI emotion prediction: {primary_emotion} ({confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in Hume AI emotion prediction: {e}")
            return {
                'success': False,
                'error': str(e),
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_emotions': [{'emotion': 'neutral', 'confidence': 0.0}],
                'model_type': 'hume_ai'
            }
        
        finally:
            # Clean up temporary file if we created it
            if temp_file and cleanup_temp:
                self._cleanup_temp_file(temp_file)
    
    def predict_emotion(self, audio_data: Union[np.ndarray, str, Path], 
                       sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Predict emotion from audio data using Hume AI (sync wrapper)
        
        Args:
            audio_data: Audio data as numpy array, file path, or Path object
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary containing emotion prediction results
        """
        try:
            # Run the async function
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.predict_emotion_async(audio_data, sample_rate))
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.predict_emotion_async(audio_data, sample_rate))
        
        except Exception as e:
            logger.error(f"Error in sync emotion prediction wrapper: {e}")
            return {
                'success': False,
                'error': str(e),
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_emotions': [{'emotion': 'neutral', 'confidence': 0.0}],
                'model_type': 'hume_ai'
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Hume AI model"""
        return {
            'model_name': 'Hume AI Emotion Recognition',
            'model_type': 'hume_ai',
            'api_endpoint': 'https://api.hume.ai/v0/batch/jobs',
            'supported_emotions': list(set(self.emotion_mapping.values())),
            'granularity': 'utterance',
            'is_trained': False,  # This is an API service, not a trained model
            'config_path': self.config_path
        }

def test_hume_emotion_recognizer():
    """Test the Hume emotion recognizer"""
    print("🧪 Testing Hume AI Emotion Recognizer")
    print("=" * 50)
    
    try:
        # Initialize recognizer
        recognizer = HumeEmotionRecognizer()
        
        # Show model info
        info = recognizer.get_model_info()
        print(f"Model: {info['model_name']}")
        print(f"Supported emotions: {info['supported_emotions']}")
        
        # Test with synthetic audio
        print("\n🎵 Testing with synthetic audio")
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create test audio (happy-sounding high frequency)
        audio = 0.3 * np.sin(2 * np.pi * 800 * t) + 0.1 * np.sin(2 * np.pi * 1200 * t)
        
        result = recognizer.predict_emotion(audio, sample_rate)
        
        if result['success']:
            print(f"✅ Prediction successful!")
            print(f"   Primary emotion: {result['emotion']} ({result['confidence']:.3f})")
            print(f"   Job ID: {result['job_id']}")
            
            print("\n📊 All emotion scores:")
            for emotion_data in result['all_emotions'][:5]:  # Show top 5
                print(f"   {emotion_data['emotion']}: {emotion_data['confidence']:.3f}")
        else:
            print(f"❌ Prediction failed: {result['error']}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hume_emotion_recognizer()
    sys.exit(0 if success else 1)