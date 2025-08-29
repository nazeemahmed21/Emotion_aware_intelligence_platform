"""
Audio utilities for recording, processing, and feature extraction
"""
import numpy as np
import librosa
import soundfile as sf
import tempfile
import logging
from typing import Tuple, Dict, Optional, Union
from pathlib import Path
import io

from config import AUDIO_CONFIG

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio recording, loading, and preprocessing"""
    
    def __init__(self, sample_rate: int = AUDIO_CONFIG["sample_rate"]):
        self.sample_rate = sample_rate
        
    def load_audio(self, audio_input: Union[str, bytes, np.ndarray]) -> Tuple[np.ndarray, int]:
        """
        Load audio from various input types
        
        Args:
            audio_input: File path, bytes, or numpy array
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            if isinstance(audio_input, str):
                # Load from file path
                y, sr = librosa.load(audio_input, sr=self.sample_rate)
                
            elif isinstance(audio_input, bytes):
                # Load from bytes (e.g., from Streamlit audio recorder)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_file.write(audio_input)
                    tmp_file.flush()
                    tmp_file.close()  # Close file before reading on Windows
                    try:
                        y, sr = librosa.load(tmp_file.name, sr=self.sample_rate)
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass
                    
            elif isinstance(audio_input, np.ndarray):
                # Already a numpy array
                y = audio_input
                sr = self.sample_rate
                
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
                
            logger.info(f"Loaded audio: {len(y)/sr:.2f}s duration, {sr}Hz sample rate")
            return y, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def preprocess_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess audio for emotion recognition
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Trim silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # Normalize
            y_normalized = librosa.util.normalize(y_trimmed)
            
            # Apply pre-emphasis filter
            y_preemphasized = librosa.effects.preemphasis(y_normalized)
            
            logger.info(f"Preprocessed audio: {len(y_preemphasized)/sr:.2f}s duration")
            return y_preemphasized
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return y
    
    def extract_features(self, y: np.ndarray, sr: int) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract comprehensive audio features for emotion recognition
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            
            # MFCCs (Mel-Frequency Cepstral Coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            features['mfcc_delta'] = np.mean(librosa.feature.delta(mfccs), axis=1)
            features['mfcc_delta2'] = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
            
            # Spectral features
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
            
            # Rhythm and tempo features
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            features['rms_energy'] = np.mean(librosa.feature.rms(y=y))
            
            # Harmonic features
            features['chroma'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            features['tonnetz'] = np.mean(librosa.feature.tonnetz(y=y, sr=sr), axis=1)
            
            # Pitch and fundamental frequency
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
            
            logger.info(f"Extracted {len(features)} feature groups")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def save_audio(self, y: np.ndarray, sr: int, filepath: str) -> bool:
        """
        Save audio to file
        
        Args:
            y: Audio time series
            sr: Sample rate
            filepath: Output file path
            
        Returns:
            Success status
        """
        try:
            sf.write(filepath, y, sr)
            logger.info(f"Audio saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False
    
    def get_audio_info(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Get basic audio information
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary with audio information
        """
        duration = len(y) / sr
        max_amplitude = np.max(np.abs(y))
        rms = np.sqrt(np.mean(y**2))
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'samples': len(y),
            'max_amplitude': max_amplitude,
            'rms_level': rms,
            'dynamic_range': 20 * np.log10(max_amplitude / (rms + 1e-10))
        }

def create_feature_vector(features: Dict[str, Union[float, np.ndarray]]) -> np.ndarray:
    """
    Convert feature dictionary to flat numpy array for ML models
    
    Args:
        features: Dictionary of features
        
    Returns:
        Flattened feature vector
    """
    feature_vector = []
    
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            feature_vector.extend(value.flatten())
        else:
            feature_vector.append(value)
    
    return np.array(feature_vector)

def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features using z-score normalization
    
    Args:
        features: Feature array
        
    Returns:
        Normalized features
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    return (features - mean) / std