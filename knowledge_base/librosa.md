2. Librosa (Audio Processing)
Overview
Librosa is a Python library for audio and music analysis. It provides the building blocks necessary to create audio information retrieval systems, including functions for loading audio, extracting features, and visualizations.

Installation
bash
pip install librosa
Core Functionality
Audio Loading
python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load an audio file
audio_path = "audio.wav"
y, sr = librosa.load(audio_path, sr=16000)  # y: audio time series, sr: sampling rate

# Get duration
duration = librosa.get_duration(y=y, sr=sr)
print(f"Audio duration: {duration:.2f} seconds")

# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('Audio Waveform')
plt.tight_layout()
plt.show()
Audio Preprocessing
python
# Trim silent parts
y_trimmed, index = librosa.effects.trim(y, top_db=20)

# Remove noise (spectral gating)
y_clean = librosa.effects.preemphasis(y)

# Resample if needed
y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)

# Normalize audio
y_normalized = librosa.util.normalize(y)
Feature Extraction for Emotion Recognition
python
# Extract Mel-Frequency Cepstral Coefficients (MFCCs)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_delta = librosa.feature.delta(mfccs)
mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

# Extract Mel-spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Extract Chroma features
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Extract Spectral features
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

# Extract Zero Crossing Rate
zcr = librosa.feature.zero_crossing_rate(y)

# Extract RMS energy
rms = librosa.feature.rms(y=y)

# Extract Tonnetz features
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

# Combine all features for emotion recognition
def extract_all_features(y, sr):
    features = {}
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    features['mfcc_std'] = np.std(mfccs, axis=1)
    
    # Spectral features
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # Other features
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['rms'] = np.mean(librosa.feature.rms(y=y))
    
    # Pitch and harmonic features
    features['chroma'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    
    return features

# Extract features from audio
audio_features = extract_all_features(y, sr)
Audio Visualization for Emotion Analysis
python
def plot_audio_features(y, sr, emotion=None):
    plt.figure(figsize=(20, 15))
    
    # Waveform
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform {f" - Emotion: {emotion}" if emotion else ""}')
    
    # Spectrogram
    plt.subplot(3, 2, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # MFCCs
    plt.subplot(3, 2, 3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    
    # Chromagram
    plt.subplot(3, 2, 4)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chromagram')
    
    # Spectral features
    plt.subplot(3, 2, 5)
    times = librosa.times_like(librosa.feature.spectral_centroid(y=y, sr=sr))
    plt.plot(times, librosa.feature.spectral_centroid(y=y, sr=sr)[0], label='Spectral Centroid')
    plt.plot(times, librosa.feature.spectral_rolloff(y=y, sr=sr)[0], label='Spectral Rolloff')
    plt.legend()
    plt.title('Spectral Features')
    
    # ZCR and RMS
    plt.subplot(3, 2, 6)
    times = librosa.times_like(zcr)
    plt.plot(times, zcr[0], label='Zero Crossing Rate')
    plt.plot(times, rms[0], label='RMS Energy')
    plt.legend()
    plt.title('Time-domain Features')
    
    plt.tight_layout()
    plt.show()

# Plot features
plot_audio_features(y, sr)
Real-time Audio Processing
python
import sounddevice as sd
import numpy as np

def audio_callback(indata, frames, time, status):
    """Callback function for real-time audio processing"""
    if status:
        print(status)
    
    # Extract features from the audio chunk
    y = indata[:, 0]  # Use first channel
    features = extract_all_features(y, sr=16000)
    
    # Process features for emotion recognition
    # (This would connect to your emotion classification model)
    process_features_for_emotion(features)

# Set up real-time audio stream
def start_realtime_analysis():
    try:
        with sd.InputStream(callback=audio_callback,
                          channels=1,
                          samplerate=16000,
                          blocksize=1024):
            print("Real-time audio analysis started. Press Enter to stop.")
            input()
    except Exception as e:
        print(f"Error: {e}")

# Helper function for emotion processing (to be implemented)
def process_features_for_emotion(features):
    """Process audio features for emotion classification"""
    # This is where you would integrate with your emotion classification model
    # For now, just print the features
    print("Processing features:", {k: np.mean(v) if isinstance(v, np.ndarray) else v 
                                  for k, v in features.items()})
Utility Functions
python
# Audio augmentation for better model training
def augment_audio(y, sr):
    augmented_versions = []
    
    # Time stretching
    y_fast = librosa.effects.time_stretch(y, rate=1.2)
    y_slow = librosa.effects.time_stretch(y, rate=0.8)
    augmented_versions.extend([y_fast, y_slow])
    
    # Pitch shifting
    y_high = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    y_low = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    augmented_versions.extend([y_high, y_low])
    
    # Add noise
    noise = np.random.randn(len(y))
    y_noisy = y + 0.005 * noise
    augmented_versions.append(y_noisy)
    
    return augmented_versions

# Save processed audio
def save_processed_audio(y, sr, filename):
    import soundfile as sf
    sf.write(filename, y, sr)
    print(f"Audio saved as {filename}")
Integration with Emotion Recognition Pipeline
python
def process_audio_for_emotion(audio_path):
    """
    Complete pipeline for audio processing for emotion recognition
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Preprocess audio
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_clean = librosa.effects.preemphasis(y_trimmed)
    
    # Extract features
    features = extract_all_features(y_clean, sr)
    
    # Normalize features (important for machine learning models)
    def normalize_features(features_dict):
        normalized = {}
        for key, value in features_dict.items():
            if isinstance(value, np.ndarray):
                # Normalize array values
                normalized[key] = (value - np.mean(value)) / np.std(value)
            else:
                # Normalize scalar values
                normalized[key] = value
        return normalized
    
    normalized_features = normalize_features(features)
    
    return normalized_features

# Example usage
audio_features = process_audio_for_emotion("audio.wav")
print("Extracted features:", audio_features)
This comprehensive Librosa documentation covers audio loading, preprocessing, feature extraction, visualization, and real-time processing - all essential for your emotion recognition pipeline.

