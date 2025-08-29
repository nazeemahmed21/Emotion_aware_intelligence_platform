"""
Main pipeline integrating all components: Audio → Transcription → Emotion → Response
"""
import logging
import os
import time
from typing import Dict, Union, Optional, List
import numpy as np

from src.audio_utils import AudioProcessor
from src.speech_to_text import get_transcriber
from src.emotion_recognition import get_emotion_recognizer
try:
    from src.emotion_recognizer_trained import TrainedEmotionRecognizer
    TRAINED_MODEL_AVAILABLE = True
except ImportError:
    TRAINED_MODEL_AVAILABLE = False

try:
    from src.emotion_recognizer_hume import HumeEmotionRecognizer
    HUME_MODEL_AVAILABLE = True
except ImportError:
    HUME_MODEL_AVAILABLE = False
from src.llm_response import get_response_generator

logger = logging.getLogger(__name__)

class EmotionAwareVoicePipeline:
    """Complete pipeline for emotion-aware voice feedback"""
    
    def __init__(self):
        """Initialize all pipeline components"""
        self.audio_processor = AudioProcessor()
        self.transcriber = None
        self.emotion_recognizer = None
        self.response_generator = None
        
        # Initialize components lazily for better startup time
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            logger.info("Initializing pipeline components...")
            
            # Initialize transcriber
            logger.info("Loading speech-to-text model...")
            self.transcriber = get_transcriber()
            
            # Initialize emotion recognizer (try Hume AI first, then trained model, then fallback)
            logger.info("Loading emotion recognition model...")
            if HUME_MODEL_AVAILABLE and os.getenv('HUME_API_KEY'):
                try:
                    self.emotion_recognizer = HumeEmotionRecognizer()
                    logger.info("✅ Using Hume AI emotion recognition")
                except Exception as e:
                    logger.warning(f"Failed to load Hume AI model: {e}")
                    # Fall back to trained model
                    if TRAINED_MODEL_AVAILABLE:
                        try:
                            self.emotion_recognizer = TrainedEmotionRecognizer()
                            if self.emotion_recognizer.is_trained_model:
                                logger.info("✅ Using trained RAVDESS emotion model")
                            else:
                                logger.info("⚠️  Trained model not found, using pretrained model")
                        except Exception as e:
                            logger.warning(f"Failed to load trained model: {e}")
                            self.emotion_recognizer = get_emotion_recognizer()
                    else:
                        self.emotion_recognizer = get_emotion_recognizer()
            elif TRAINED_MODEL_AVAILABLE:
                try:
                    self.emotion_recognizer = TrainedEmotionRecognizer()
                    if self.emotion_recognizer.is_trained_model:
                        logger.info("✅ Using trained RAVDESS emotion model")
                    else:
                        logger.info("⚠️  Trained model not found, using pretrained model")
                except Exception as e:
                    logger.warning(f"Failed to load trained model: {e}")
                    self.emotion_recognizer = get_emotion_recognizer()
            else:
                self.emotion_recognizer = get_emotion_recognizer()
            
            # Initialize response generator
            logger.info("Initializing response generator...")
            self.response_generator = get_response_generator()
            
            logger.info("Pipeline initialization complete!")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    def process_audio(self, 
                     audio_input: Union[str, bytes, np.ndarray],
                     include_features: bool = False,
                     context: Optional[str] = None) -> Dict:
        """
        Process audio through complete pipeline
        
        Args:
            audio_input: Audio file path, bytes, or numpy array
            include_features: Whether to include audio features in output
            context: Additional context for response generation
            
        Returns:
            Complete processing results
        """
        start_time = time.time()
        result = {
            'success': False,
            'timestamp': time.time(),
            'processing_time': 0.0,
            'steps': {}
        }
        
        try:
            logger.info("Starting audio processing pipeline...")
            
            # Step 1: Load and preprocess audio
            step_start = time.time()
            logger.info("Step 1: Loading and preprocessing audio...")
            
            y, sr = self.audio_processor.load_audio(audio_input)
            y_processed = self.audio_processor.preprocess_audio(y, sr)
            audio_info = self.audio_processor.get_audio_info(y_processed, sr)
            
            result['steps']['audio_processing'] = {
                'duration': time.time() - step_start,
                'audio_info': audio_info,
                'success': True
            }
            
            # Step 2: Speech-to-text transcription
            step_start = time.time()
            logger.info("Step 2: Transcribing speech...")
            
            transcription_result = self.transcriber.transcribe_audio(y_processed)
            transcription = transcription_result.get('text', '').strip()
            
            if not transcription:
                logger.warning("No transcription obtained - continuing with emotion analysis")
                transcription = "[No speech detected]"  # Placeholder for display
                result['steps']['transcription'] = {
                    'duration': time.time() - step_start,
                    'text': transcription,
                    'confidence': 0.0,
                    'language': 'unknown',
                    'success': False,
                    'error': 'No speech detected'
                }
            else:
                result['steps']['transcription'] = {
                    'duration': time.time() - step_start,
                    'text': transcription,
                    'confidence': transcription_result.get('confidence', 0.0),
                    'language': transcription_result.get('language', 'en'),
                    'success': True
                }
            
            # Step 3: Emotion recognition
            step_start = time.time()
            logger.info("Step 3: Recognizing emotion...")
            
            emotion_result = self.emotion_recognizer.predict_emotion(y_processed)
            emotion = emotion_result.get('emotion', 'neutral')
            emotion_confidence = emotion_result.get('confidence', 0.5)
            
            result['steps']['emotion_recognition'] = {
                'duration': time.time() - step_start,
                'emotion': emotion,
                'confidence': emotion_confidence,
                'all_emotions': emotion_result.get('all_emotions', []),
                'emotion_scores': emotion_result.get('emotion_scores', {}),
                'aggregated_analysis': emotion_result.get('aggregated_analysis', {}),
                'method': emotion_result.get('method', 'unknown'),
                'model_type': emotion_result.get('model_type', 'unknown'),
                'success': True
            }
            
            # Step 4: Generate empathetic response
            step_start = time.time()
            logger.info("Step 4: Generating empathetic response...")
            
            # Only generate response if we have valid transcription
            transcription_success = result['steps']['transcription']['success']
            if transcription_success:
                response_result = self.response_generator.generate_empathetic_response(
                    transcription, emotion, emotion_confidence, context
                )
                response_text = response_result.get('response', '')
            else:
                # Generate emotion-only response when no transcription available
                response_text = f"I can sense a {emotion} emotional tone in your audio, though I couldn't make out the specific words. The emotional intensity appears to be {emotion_confidence:.1%}."
                response_result = {'response': response_text, 'model': 'emotion_only'}
            
            result['steps']['response_generation'] = {
                'duration': time.time() - step_start,
                'response': response_text,
                'model': response_result.get('model', 'unknown'),
                'success': True
            }
            
            # Step 5: Extract features if requested
            if include_features:
                step_start = time.time()
                logger.info("Step 5: Extracting audio features...")
                
                features = self.audio_processor.extract_features(y_processed, sr)
                result['steps']['feature_extraction'] = {
                    'duration': time.time() - step_start,
                    'features': features,
                    'success': True
                }
            
            # Compile final results
            total_time = time.time() - start_time
            result.update({
                'success': True,
                'processing_time': total_time,
                'transcription': transcription,
                'emotion': emotion,
                'emotion_confidence': emotion_confidence,
                'response': response_result.get('response', ''),
                'audio_duration': audio_info.get('duration', 0.0)
            })
            
            logger.info(f"Pipeline completed successfully in {total_time:.2f}s")
            logger.info(f"Result: '{transcription}' → {emotion} ({emotion_confidence:.2f}) → '{result['response'][:50]}...'")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.update({
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            })
            return result
    
    def process_batch(self, 
                     audio_inputs: List[Union[str, bytes, np.ndarray]],
                     include_features: bool = False) -> List[Dict]:
        """
        Process multiple audio inputs
        
        Args:
            audio_inputs: List of audio inputs
            include_features: Whether to include features
            
        Returns:
            List of processing results
        """
        results = []
        total_start = time.time()
        
        logger.info(f"Processing batch of {len(audio_inputs)} audio files...")
        
        for i, audio_input in enumerate(audio_inputs):
            logger.info(f"Processing audio {i+1}/{len(audio_inputs)}")
            result = self.process_audio(audio_input, include_features)
            result['batch_index'] = i
            results.append(result)
        
        total_time = time.time() - total_start
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        
        return results
    
    def get_pipeline_status(self) -> Dict:
        """Get status of all pipeline components"""
        status = {
            'audio_processor': self.audio_processor is not None,
            'transcriber': self.transcriber is not None,
            'emotion_recognizer': self.emotion_recognizer is not None,
            'response_generator': self.response_generator is not None,
            'ready': False
        }
        
        status['ready'] = all(status.values())
        
        # Additional component-specific status
        if self.transcriber:
            status['transcriber_model'] = getattr(self.transcriber, 'model_size', 'unknown')
        
        if self.emotion_recognizer:
            status['emotion_model'] = getattr(self.emotion_recognizer, 'model_name', 'unknown')
        
        if self.response_generator:
            status['llm_model'] = getattr(self.response_generator, 'model_name', 'unknown')
            status['conversation_history'] = len(getattr(self.response_generator, 'conversation_history', []))
        
        return status
    
    def clear_conversation_history(self):
        """Clear conversation history from response generator"""
        if self.response_generator:
            self.response_generator.clear_history()
            logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict:
        """Get conversation summary"""
        if self.response_generator:
            return self.response_generator.get_conversation_summary()
        return {'message_count': 0, 'emotions': [], 'duration': 0}

class StreamingPipeline:
    """Streaming version of the pipeline for real-time processing"""
    
    def __init__(self, chunk_duration: float = 3.0):
        """
        Initialize streaming pipeline
        
        Args:
            chunk_duration: Duration of audio chunks in seconds
        """
        self.chunk_duration = chunk_duration
        self.pipeline = EmotionAwareVoicePipeline()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.sample_rate = 16000
        self.is_streaming = False
        
    def start_streaming(self):
        """Start streaming mode"""
        self.is_streaming = True
        self.audio_buffer = np.array([], dtype=np.float32)
        logger.info("Streaming pipeline started")
    
    def stop_streaming(self):
        """Stop streaming mode"""
        self.is_streaming = False
        logger.info("Streaming pipeline stopped")
    
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict]:
        """
        Add audio chunk and process if buffer is full
        
        Args:
            audio_chunk: New audio data
            
        Returns:
            Processing result if chunk is ready, None otherwise
        """
        if not self.is_streaming:
            return None
        
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Check if we have enough audio
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        
        if buffer_duration >= self.chunk_duration:
            # Process the chunk
            chunk_samples = int(self.chunk_duration * self.sample_rate)
            audio_to_process = self.audio_buffer[:chunk_samples]
            
            # Keep remaining audio for next chunk
            self.audio_buffer = self.audio_buffer[chunk_samples:]
            
            # Process the chunk
            result = self.pipeline.process_audio(audio_to_process)
            result['chunk_info'] = {
                'duration': self.chunk_duration,
                'buffer_remaining': len(self.audio_buffer) / self.sample_rate
            }
            
            return result
        
        return None

# Convenience functions
def process_audio_file(file_path: str, include_features: bool = False) -> Dict:
    """
    Quick processing of audio file
    
    Args:
        file_path: Path to audio file
        include_features: Include audio features
        
    Returns:
        Processing result
    """
    pipeline = EmotionAwareVoicePipeline()
    return pipeline.process_audio(file_path, include_features)

def process_audio_bytes(audio_bytes: bytes, include_features: bool = False) -> Dict:
    """
    Quick processing of audio bytes
    
    Args:
        audio_bytes: Audio data as bytes
        include_features: Include audio features
        
    Returns:
        Processing result
    """
    pipeline = EmotionAwareVoicePipeline()
    return pipeline.process_audio(audio_bytes, include_features)

# Global pipeline instance
_global_pipeline = None

def get_pipeline() -> EmotionAwareVoicePipeline:
    """
    Get global pipeline instance (singleton pattern)
    
    Returns:
        EmotionAwareVoicePipeline instance
    """
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = EmotionAwareVoicePipeline()
    return _global_pipeline