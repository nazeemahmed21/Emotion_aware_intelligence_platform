"""
Audio quality validation and preprocessing
"""
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class AudioValidator:
    """Validates and preprocesses audio for better transcription quality"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def validate_and_preprocess(self, audio: np.ndarray, 
                               sample_rate: int = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Validate and preprocess audio for optimal transcription
        
        Args:
            audio: Audio array
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (processed_audio, validation_info)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        validation_info = {
            'original_length': len(audio),
            'original_duration': len(audio) / sample_rate,
            'issues': [],
            'fixes_applied': [],
            'quality_score': 0.0
        }
        
        try:
            # Check if audio is valid
            if len(audio) == 0:
                validation_info['issues'].append('Empty audio')
                return None, validation_info
            
            # Check duration
            duration = len(audio) / sample_rate
            if duration < 0.5:
                validation_info['issues'].append(f'Too short: {duration:.2f}s (minimum 0.5s)')
            elif duration > 30:
                validation_info['issues'].append(f'Too long: {duration:.2f}s (maximum 30s)')
                # Trim to 30 seconds
                audio = audio[:30 * sample_rate]
                validation_info['fixes_applied'].append('Trimmed to 30 seconds')
            
            # Check for silence
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.001:
                validation_info['issues'].append(f'Too quiet: RMS={rms:.6f}')
                return None, validation_info
            
            # Check for clipping
            clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
            if clipping_ratio > 0.01:
                validation_info['issues'].append(f'Audio clipping detected: {clipping_ratio:.1%}')
            
            # Normalize audio
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
                validation_info['fixes_applied'].append('Normalized amplitude')
            
            # Remove DC offset
            audio = audio - np.mean(audio)
            validation_info['fixes_applied'].append('Removed DC offset')
            
            # Apply gentle high-pass filter to remove low-frequency noise
            audio = self._apply_highpass_filter(audio, sample_rate)
            validation_info['fixes_applied'].append('Applied high-pass filter')
            
            # Ensure final audio is contiguous and correct type
            audio = np.ascontiguousarray(audio.astype(np.float32))
            
            # Check for repetitive patterns (like "FAAA FAAA")
            repetition_score = self._detect_repetitive_patterns(audio)
            if repetition_score > 0.8:
                validation_info['issues'].append(f'Highly repetitive audio detected: {repetition_score:.2f}')
            
            # Calculate quality score
            validation_info['quality_score'] = self._calculate_quality_score(audio, sample_rate)
            
            # Final validation
            if validation_info['quality_score'] < 0.3:
                validation_info['issues'].append(f'Low quality score: {validation_info["quality_score"]:.2f}')
            
            validation_info['processed_length'] = len(audio)
            validation_info['processed_duration'] = len(audio) / sample_rate
            
            return audio, validation_info
            
        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            validation_info['issues'].append(f'Validation error: {str(e)}')
            return None, validation_info
    
    def _apply_highpass_filter(self, audio: np.ndarray, sample_rate: int, 
                              cutoff: float = 80.0) -> np.ndarray:
        """Apply simple high-pass filter to remove low-frequency noise"""
        try:
            from scipy import signal
            
            # Design high-pass filter
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            b, a = signal.butter(2, normalized_cutoff, btype='high')
            
            # Apply filter and ensure contiguous array
            filtered_audio = signal.filtfilt(b, a, audio)
            
            # Ensure the result is contiguous to avoid stride issues
            if not filtered_audio.flags['C_CONTIGUOUS']:
                filtered_audio = np.ascontiguousarray(filtered_audio)
            
            return filtered_audio.astype(np.float32)
            
        except ImportError:
            # Fallback: simple high-pass using difference
            logger.warning("SciPy not available, using simple high-pass filter")
            alpha = 0.95
            filtered = np.zeros_like(audio, dtype=np.float32)
            filtered[0] = audio[0]
            for i in range(1, len(audio)):
                filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])
            return filtered
        except Exception as e:
            logger.warning(f"High-pass filter failed: {e}")
            # Return original audio as contiguous array
            return np.ascontiguousarray(audio.astype(np.float32))
    
    def _detect_repetitive_patterns(self, audio: np.ndarray) -> float:
        """Detect repetitive patterns in audio that might cause transcription issues"""
        try:
            # Calculate autocorrelation
            correlation = np.correlate(audio, audio, mode='full')
            correlation = correlation[len(correlation)//2:]
            
            # Normalize
            correlation = correlation / correlation[0]
            
            # Look for strong periodic patterns
            # Skip the first few samples to avoid trivial correlation
            start_idx = int(0.01 * len(correlation))  # Skip first 1%
            end_idx = int(0.5 * len(correlation))     # Look at first 50%
            
            if end_idx > start_idx:
                max_correlation = np.max(correlation[start_idx:end_idx])
                return max_correlation
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
            return 0.0
    
    def _calculate_quality_score(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate overall audio quality score (0-1)"""
        try:
            score = 1.0
            
            # Check RMS level
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.01:
                score *= 0.5  # Too quiet
            elif rms > 0.5:
                score *= 0.8  # Too loud
            
            # Check dynamic range
            dynamic_range = np.max(audio) - np.min(audio)
            if dynamic_range < 0.1:
                score *= 0.6  # Too little variation
            
            # Check for clipping
            clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
            score *= (1.0 - clipping_ratio)
            
            # Check spectral content (simple measure)
            # Higher frequencies usually indicate speech
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
            
            # Energy in speech frequency range (300-3400 Hz)
            speech_mask = (freqs >= 300) & (freqs <= 3400)
            speech_energy = np.sum(np.abs(fft[speech_mask])**2)
            total_energy = np.sum(np.abs(fft)**2)
            
            if total_energy > 0:
                speech_ratio = speech_energy / total_energy
                score *= (0.5 + 0.5 * speech_ratio)  # Boost if speech-like
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.5  # Default moderate score

def validate_audio_for_transcription(audio: np.ndarray, 
                                   sample_rate: int = 16000) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Convenience function to validate audio for transcription
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        
    Returns:
        Tuple of (processed_audio, validation_info)
    """
    validator = AudioValidator(sample_rate)
    return validator.validate_and_preprocess(audio, sample_rate)