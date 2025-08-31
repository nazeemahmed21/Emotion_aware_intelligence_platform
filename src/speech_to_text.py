"""
Speech-to-text transcription using Faster Whisper
"""
from faster_whisper import WhisperModel
import numpy as np
import tempfile
import logging
from typing import Union, Dict, Optional
import os

from config import WHISPER_CONFIG

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Handles speech-to-text transcription using Faster Whisper"""
    
    def __init__(self, model_size: str = WHISPER_CONFIG["model_size"]):
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Whisper model"""
        try:
            logger.info(f"Loading Faster Whisper model: {self.model_size}")
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            logger.info("Faster Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Faster Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_input: Union[str, bytes, np.ndarray], 
                        language: str = WHISPER_CONFIG["language"]) -> Dict[str, Union[str, float]]:
        """
        Transcribe audio to text
        
        Args:
            audio_input: Audio file path, bytes, or numpy array
            language: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # Convert all inputs to numpy array first
            audio_array = None
            sample_rate = 16000  # Default sample rate
            
            if isinstance(audio_input, bytes):
                # Convert bytes to numpy array
                import soundfile as sf
                import io
                
                try:
                    # Try to read as audio file from bytes
                    audio_array, sample_rate = sf.read(io.BytesIO(audio_input))
                    logger.info(f"Loaded audio from bytes: {len(audio_array)} samples at {sample_rate}Hz")
                    
                except Exception as e:
                    logger.warning(f"Could not read bytes as audio file: {e}")
                    # Try different interpretations of the bytes
                    try:
                        # First try: assume it's WAV format but read differently
                        import wave
                        with io.BytesIO(audio_input) as audio_io:
                            with wave.open(audio_io, 'rb') as wav_file:
                                frames = wav_file.readframes(-1)
                                sample_rate = wav_file.getframerate()
                                channels = wav_file.getnchannels()
                                sample_width = wav_file.getsampwidth()
                                
                                # Convert to numpy array
                                if sample_width == 1:
                                    audio_array = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                                elif sample_width == 2:
                                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                                elif sample_width == 4:
                                    audio_array = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                                else:
                                    raise ValueError(f"Unsupported sample width: {sample_width}")
                                
                                # Handle stereo
                                if channels == 2:
                                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                                
                                logger.info(f"Loaded WAV from bytes: {len(audio_array)} samples at {sample_rate}Hz")
                                
                    except Exception as e2:
                        logger.warning(f"WAV parsing failed: {e2}")
                        # Final fallback: assume raw 16-bit PCM data
                        try:
                            audio_array = np.frombuffer(audio_input, dtype=np.int16).astype(np.float32) / 32768.0
                            sample_rate = 16000
                            logger.info(f"Interpreted bytes as raw PCM: {len(audio_array)} samples")
                        except Exception as e3:
                            logger.error(f"All audio conversion methods failed: {e3}")
                            raise ValueError("Cannot process audio bytes")
                    
            elif isinstance(audio_input, np.ndarray):
                # Already a numpy array
                audio_array = audio_input.copy()
                sample_rate = 16000  # Assume 16kHz
                logger.info(f"Using numpy array: {len(audio_array)} samples")
                
            elif isinstance(audio_input, str):
                # Load from file path using librosa (more reliable than Whisper's loader)
                try:
                    import librosa
                    audio_array, sample_rate = librosa.load(audio_input, sr=16000)
                    logger.info(f"Loaded from file: {len(audio_array)} samples at {sample_rate}Hz")
                except ImportError:
                    # Fallback to direct file transcription if librosa is not available
                    logger.info("Librosa not available, using direct file transcription")
                    # Pass the file path directly to Faster Whisper
                    audio_array = audio_input
                    sample_rate = 16000
                
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
            
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
                logger.info("Converted to mono")
            
            # Resample to 16kHz if needed (Whisper's expected sample rate)
            if sample_rate != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
                logger.info("Resampled to 16kHz")
            
            # Normalize audio
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array)) * 0.95
            
            # Ensure proper format for Whisper
            audio_array = np.ascontiguousarray(audio_array.astype(np.float32))
            
            # Use Faster Whisper's transcribe method
            logger.info("Starting transcription...")
            result = self.model.transcribe(
                audio_array,  # Pass numpy array directly
                language=language,
                task=WHISPER_CONFIG["task"],
                temperature=0.0,
                beam_size=1
            )
            
            # Faster Whisper returns a tuple: (segments_generator, info)
            segments = list(result[0])  # Convert generator to list
            info = result[1]  # TranscriptionInfo object
            
            # Extract text from segments
            text = " ".join([seg.text for seg in segments if hasattr(seg, 'text')])
            
            # Extract key information from Faster Whisper result
            transcription_result = {
                'text': text.strip(),
                'language': info.language or language,
                'confidence': self._calculate_confidence(segments),
                'segments': [{'start': seg.start, 'end': seg.end, 'text': seg.text} for seg in segments if hasattr(seg, 'start')],
                'duration': info.duration
            }
            
            logger.info(f"Transcription completed: '{transcription_result['text'][:50]}...'")
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return {
                'text': '',
                'language': language,
                'confidence': 0.0,
                'segments': [],
                'duration': 0.0,
                'error': str(e)
            }
    
    def _calculate_confidence(self, segments) -> float:
        """
        Calculate average confidence from segments
        
        Args:
            segments: List of Faster Whisper segments
            
        Returns:
            Average confidence score
        """
        try:
            if not segments:
                return 0.8  # Default confidence if no segments
            
            # Faster Whisper provides avg_logprob for each segment
            confidences = []
            for segment in segments:
                if hasattr(segment, 'avg_logprob') and segment.avg_logprob is not None:
                    # Convert log probability to confidence (rough approximation)
                    confidence = min(1.0, max(0.0, np.exp(segment.avg_logprob)))
                    confidences.append(confidence)
            
            if confidences:
                return np.mean(confidences)
            else:
                return 0.8  # Default confidence
                
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.8
    

    
    def transcribe_with_timestamps(self, audio_input: Union[str, bytes, np.ndarray]) -> Dict:
        """
        Transcribe audio with detailed word-level timestamps
        
        Args:
            audio_input: Audio input
            
        Returns:
            Detailed transcription with timestamps
        """
        result = self.transcribe_audio(audio_input)
        
        if 'error' in result:
            return result
        
        # Extract word-level timestamps if available
        words_with_timestamps = []
        for segment in result.get('segments', []):
            # Faster Whisper doesn't provide word-level timestamps by default
            # We can only provide segment-level timestamps
            words_with_timestamps.append({
                'segment': segment.get('text', '').strip(),
                'start': segment.get('start', 0.0),
                'end': segment.get('end', 0.0),
                'confidence': 0.8  # Default confidence
            })
        
        result['words'] = words_with_timestamps
        return result
    
    def is_speech_detected(self, audio_input: Union[str, bytes, np.ndarray], 
                          min_speech_duration: float = 0.5) -> bool:
        """
        Check if speech is detected in audio
        
        Args:
            audio_input: Audio input
            min_speech_duration: Minimum duration to consider as speech
            
        Returns:
            True if speech is detected
        """
        try:
            result = self.transcribe_audio(audio_input)
            
            # Check if transcription is meaningful
            text = result.get('text', '').strip()
            duration = result.get('duration', 0.0)
            confidence = result.get('confidence', 0.0)
            
            # Criteria for speech detection
            has_text = len(text) > 0 and not text.lower() in ['', 'you', 'thank you', '.', '...']
            sufficient_duration = duration >= min_speech_duration
            sufficient_confidence = confidence >= 0.3
            
            return has_text and sufficient_duration and sufficient_confidence
            
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            return False

# Convenience function for quick transcription
def transcribe_audio_file(file_path: str, model_size: str = "base") -> str:
    """
    Quick transcription of audio file
    
    Args:
        file_path: Path to audio file
        model_size: Faster Whisper model size
        
    Returns:
        Transcribed text
    """
    transcriber = WhisperTranscriber(model_size)
    result = transcriber.transcribe_audio(file_path)
    return result.get('text', '')

# Global transcriber instance for efficiency
_global_transcriber = None

def get_transcriber(model_size: str = WHISPER_CONFIG["model_size"]) -> WhisperTranscriber:
    """
    Get global transcriber instance (singleton pattern)
    
    Args:
        model_size: Faster Whisper model size
        
    Returns:
        WhisperTranscriber instance
    """
    global _global_transcriber
    if _global_transcriber is None or _global_transcriber.model_size != model_size:
        _global_transcriber = WhisperTranscriber(model_size)
    return _global_transcriber