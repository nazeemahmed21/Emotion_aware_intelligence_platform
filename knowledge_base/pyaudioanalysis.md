3. PyAudioAnalysis for Emotion Recognition
Overview
pyAudioAnalysis is a Python library covering a wide range of audio analysis tasks. It provides functionality for feature extraction, classification, segmentation, and visualization. For emotion recognition, we'll use its feature extraction capabilities combined with machine learning models.

Installation
bash
pip install pyAudioAnalysis
Core Functionality
Basic Feature Extraction
python
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np

# Read audio file
[fs, x] = audioBasicIO.read_audio_file("audio.wav")

# Convert to mono if stereo
if len(x.shape) > 1:
    x = np.mean(x, axis=1)

# Extract short-term features
F, f_names = audioFeatureExtraction.stFeatureExtraction(x, fs, 0.050*fs, 0.025*fs)

print(f"Extracted {F.shape[0]} features with {F.shape[1]} frames")
print("Feature names:", f_names)
Comprehensive Feature Extraction Function
python
def extract_audio_features(audio_path):
    """
    Extract comprehensive audio features for emotion recognition
    """
    # Read audio file
    [fs, x] = audioBasicIO.read_audio_file(audio_path)
    
    # Convert to mono if stereo
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)
    
    # Short-term feature extraction
    win_size = 0.050 * fs  # 50ms window
    step_size = 0.025 * fs  # 25ms step
    
    # Extract features
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, fs, win_size, step_size)
    
    # Compute statistics for each feature
    features = {}
    for i, name in enumerate(f_names):
        features[f"{name}_mean"] = np.mean(F[i, :])
        features[f"{name}_std"] = np.std(F[i, :])
        features[f"{name}_max"] = np.max(F[i, :])
        features[f"{name}_min"] = np.min(F[i, :])
    
    # Additional mid-term features
    mid_term_size = 1.0 * fs  # 1-second segments
    mid_term_step = 0.5 * fs  # 0.5-second step
    
    mt_features, mt_feature_names, _ = audioFeatureExtraction.mtFeatureExtraction(
        x, fs, mid_term_size, mid_term_step, win_size, step_size
    )
    
    # Add mid-term features statistics
    for i, name in enumerate(mt_feature_names):
        features[f"mt_{name}_mean"] = np.mean(mt_features[i, :])
        features[f"mt_{name}_std"] = np.std(mt_features[i, :])
    
    return features, F, f_names

# Extract features
features, F, feature_names = extract_audio_features("audio.wav")
print(f"Extracted {len(features)} features")
Emotion Classification Setup
python
from pyAudioAnalysis import audioTrainTest as aT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# Prepare dataset for emotion recognition
def prepare_emotion_dataset(data_folder, emotions):
    """
    Prepare dataset from folder structure: data_folder/emotion/*.wav
    """
    X = []  # Features
    y = []  # Labels
    filenames = []  # Audio filenames
    
    for emotion in emotions:
        emotion_folder = os.path.join(data_folder, emotion)
        
        if not os.path.exists(emotion_folder):
            print(f"Warning: Folder {emotion_folder} does not exist")
            continue
            
        for audio_file in os.listdir(emotion_folder):
            if audio_file.endswith(('.wav', '.mp3', '.m4a')):
                audio_path = os.path.join(emotion_folder, audio_file)
                
                try:
                    # Extract features
                    features, _, _ = extract_audio_features(audio_path)
                    
                    # Convert to array (preserve order)
                    feature_vector = list(features.values())
                    
                    X.append(feature_vector)
                    y.append(emotion)
                    filenames.append(audio_file)
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
    
    return np.array(X), np.array(y), filenames

# Example emotions (modify based on your dataset)
emotions = ["happy", "sad", "angry", "neutral", "excited", "calm"]
X, y, filenames = prepare_emotion_dataset("emotional_speech_data", emotions)
Train Emotion Classification Model
python
def train_emotion_classifier(X, y, test_size=0.2, model_type="svm"):
    """
    Train emotion classification model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model using pyAudioAnalysis
    model_path = "emotion_classifier"
    
    # Create directory structure expected by pyAudioAnalysis
    os.makedirs("training_data", exist_ok=True)
    for emotion in set(y):
        os.makedirs(f"training_data/{emotion}", exist_ok=True)
    
    # Note: In practice, you'd need to organize your audio files in this structure
    # and use aT.extract_features_and_train() directly
    
    # Alternative: Use sklearn directly
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    
    if model_type == "svm":
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model: {model_type}")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler, accuracy

# Train the model
model, scaler, accuracy = train_emotion_classifier(X, y, model_type="svm")
Real-time Emotion Recognition
python
import pyaudio
import numpy as np
import time

class RealTimeEmotionAnalyzer:
    def __init__(self, model, scaler, sample_rate=16000, chunk_size=1024):
        self.model = model
        self.scaler = scaler
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_duration = 3.0  # seconds
        self.buffer_size = int(self.buffer_duration * sample_rate)
        
        # Initialize audio stream
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
            stream_callback=self.audio_callback
        )
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        # Convert bytes to numpy array
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Keep only the most recent audio
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
        
        return (in_data, pyaudio.paContinue)
    
    def extract_features_from_buffer(self):
        if len(self.audio_buffer) < self.sample_rate:  # Need at least 1 second
            return None
        
        # Extract features using pyAudioAnalysis
        win_size = int(0.050 * self.sample_rate)
        step_size = int(0.025 * self.sample_rate)
        
        try:
            F, f_names = audioFeatureExtraction.stFeatureExtraction(
                self.audio_buffer, self.sample_rate, win_size, step_size
            )
            
            # Compute statistics
            features = {}
            for i, name in enumerate(f_names):
                features[f"{name}_mean"] = np.mean(F[i, :])
                features[f"{name}_std"] = np.std(F[i, :])
            
            # Convert to feature vector (same order as training)
            feature_vector = np.array([features.get(f"{name}_mean", 0) for name in f_names] + 
                                     [features.get(f"{name}_std", 0) for name in f_names])
            
            return feature_vector.reshape(1, -1)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def predict_emotion(self):
        feature_vector = self.extract_features_from_buffer()
        if feature_vector is not None:
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Predict emotion
            emotion = self.model.predict(feature_vector_scaled)[0]
            confidence = np.max(self.model.predict_proba(feature_vector_scaled))
            
            return emotion, confidence
        return None, 0.0
    
    def start_analysis(self):
        print("Starting real-time emotion analysis...")
        try:
            self.stream.start_stream()
            
            while self.stream.is_active():
                emotion, confidence = self.predict_emotion()
                if emotion:
                    print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
                time.sleep(1.0)  # Predict every second
                
        except KeyboardInterrupt:
            print("Stopping analysis...")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

# Initialize and start analyzer
# analyzer = RealTimeEmotionAnalyzer(model, scaler)
# analyzer.start_analysis()
Visualization of Audio Features
python
def visualize_audio_features(audio_path):
    """
    Visualize audio features for emotion analysis
    """
    # Read audio
    [fs, x] = audioBasicIO.read_audio_file(audio_path)
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)
    
    # Extract features
    win_size = int(0.050 * fs)
    step_size = int(0.025 * fs)
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, fs, win_size, step_size)
    
    # Create visualization
    plt.figure(figsize=(15, 12))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    time_axis = np.arange(len(x)) / float(fs)
    plt.plot(time_axis, x)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Plot MFCCs (if available)
    mfcc_indices = [i for i, name in enumerate(f_names) if 'mfcc' in name]
    if mfcc_indices:
        plt.subplot(3, 1, 2)
        mfccs = F[mfcc_indices, :]
        plt.imshow(mfccs, aspect='auto', origin='lower', 
                  extent=[0, len(x)/fs, 0, len(mfcc_indices)])
        plt.title("MFCC Features")
        plt.xlabel("Time (s)")
        plt.ylabel("MFCC Coefficient")
        plt.colorbar()
    
    # Plot energy and spectral features
    plt.subplot(3, 1, 3)
    time_axis_features = np.arange(F.shape[1]) * step_size / fs
    
    # Plot energy
    energy_idx = f_names.index('energy')
    plt.plot(time_axis_features, F[energy_idx, :], label='Energy', alpha=0.7)
    
    # Plot spectral centroid if available
    if 'spectral_centroid' in f_names:
        centroid_idx = f_names.index('spectral_centroid')
        plt.plot(time_axis_features, F[centroid_idx, :], label='Spectral Centroid', alpha=0.7)
    
    plt.title("Energy and Spectral Features")
    plt.xlabel("Time (s)")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualize features
visualize_audio_features("audio.wav")
Saving and Loading Models
python
import joblib

def save_emotion_model(model, scaler, feature_names, model_path="emotion_model.pkl"):
    """
    Save trained emotion recognition model
    """
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'timestamp': time.time()
    }
    
    joblib.dump(model_data, model_path)
    print(f"Model saved to {model_path}")

def load_emotion_model(model_path="emotion_model.pkl"):
    """
    Load trained emotion recognition model
    """
    try:
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Save model
save_emotion_model(model, scaler, feature_names)

# Load model
loaded_model_data = load_emotion_model()
if loaded_model_data:
    model = loaded_model_data['model']
    scaler = loaded_model_data['scaler']
    feature_names = loaded_model_data['feature_names']
This comprehensive pyAudioAnalysis documentation covers feature extraction, model training, real-time emotion recognition, and visualization - everything you need for your emotion-aware voice feedback bot.

