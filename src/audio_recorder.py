"""
Real-time audio recording component for Streamlit
"""
import streamlit as st
import numpy as np
import threading
import time
import queue
import logging
from typing import Optional, Callable, Dict, Any
import tempfile
import os

logger = logging.getLogger(__name__)

class StreamlitAudioRecorder:
    """Real-time audio recorder for Streamlit applications"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024):
        """
        Initialize audio recorder
        
        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
            chunk_size: Audio chunk size for processing
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.is_recording = False
        self.audio_data = []
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        
        # Try to import audio libraries
        self.audio_available = self._check_audio_libraries()
    
    def _check_audio_libraries(self) -> bool:
        """Check if audio recording libraries are available"""
        try:
            import pyaudio
            self.pyaudio = pyaudio
            return True
        except ImportError:
            logger.warning("PyAudio not available. Audio recording disabled.")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to queue for processing
        if not self.audio_queue.full():
            self.audio_queue.put(audio_chunk)
        
        # Store for final recording
        self.audio_data.append(audio_chunk)
        
        return (in_data, self.pyaudio.paContinue if self.is_recording else self.pyaudio.paComplete)
    
    def start_recording(self) -> bool:
        """Start audio recording"""
        if not self.audio_available:
            logger.error("Audio recording not available")
            return False
        
        if self.is_recording:
            logger.warning("Already recording")
            return False
        
        try:
            # Initialize PyAudio
            p = self.pyaudio.PyAudio()
            
            # Open audio stream
            self.stream = p.open(
                format=self.pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Clear previous data
            self.audio_data = []
            while not self.audio_queue.empty():
                self.audio_queue.get()
            
            # Start recording
            self.is_recording = True
            self.stream.start_stream()
            
            logger.info("Audio recording started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """Stop audio recording and return recorded data"""
        if not self.is_recording:
            logger.warning("Not currently recording")
            return None
        
        try:
            # Stop recording
            self.is_recording = False
            
            # Wait a bit for the stream to finish
            time.sleep(0.1)
            
            # Stop and close stream
            if hasattr(self, 'stream'):
                self.stream.stop_stream()
                self.stream.close()
            
            # Combine all audio data
            if self.audio_data:
                combined_audio = np.concatenate(self.audio_data)
                logger.info(f"Recording stopped. Duration: {len(combined_audio) / self.sample_rate:.2f}s")
                return combined_audio
            else:
                logger.warning("No audio data recorded")
                return None
                
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return None
    
    def get_recording_level(self) -> float:
        """Get current recording level (0.0 to 1.0)"""
        if not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                rms = np.sqrt(np.mean(chunk**2))
                return min(1.0, rms * 10)  # Scale for visualization
            except queue.Empty:
                pass
        return 0.0
    
    def is_recording_active(self) -> bool:
        """Check if recording is active"""
        return self.is_recording

class WebAudioRecorder:
    """Web-based audio recorder using Streamlit components"""
    
    def __init__(self):
        """Initialize web audio recorder"""
        self.recorded_audio = None
        self.is_recording = False
    
    def render_recorder_ui(self) -> Optional[bytes]:
        """Render web-based audio recorder UI"""
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                🎙️ Voice Recording
            </div>
            <div style="text-align: center; padding: 1rem;">
                <p>Click the button below to start recording your voice</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Try to use streamlit-webrtc if available
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
            import av
            
            # WebRTC configuration
            rtc_configuration = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })
            
            # Audio recorder
            webrtc_ctx = webrtc_streamer(
                key="audio-recorder",
                mode=WebRtcMode.SENDONLY,
                audio_receiver_size=1024,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": False, "audio": True},
            )
            
            if webrtc_ctx.audio_receiver:
                # Process audio frames
                audio_frames = []
                while True:
                    try:
                        audio_frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                        audio_frames.append(audio_frame)
                    except queue.Empty:
                        break
                
                if audio_frames:
                    # Convert frames to numpy array
                    audio_data = []
                    for frame in audio_frames:
                        sound = frame.to_ndarray()
                        audio_data.append(sound)
                    
                    if audio_data:
                        combined_audio = np.concatenate(audio_data, axis=0)
                        return combined_audio.tobytes()
            
            return None
            
        except ImportError:
            # Fallback to file upload
            st.info("📱 Real-time recording requires additional setup. Using file upload instead.")
            
            uploaded_file = st.file_uploader(
                "Upload an audio file",
                type=['wav', 'mp3', 'm4a', 'ogg'],
                help="Upload an audio file to analyze emotions"
            )
            
            if uploaded_file is not None:
                return uploaded_file.read()
            
            return None

class AudioRecorderComponent:
    """Main audio recorder component with fallback options"""
    
    def __init__(self):
        """Initialize audio recorder with multiple fallback options"""
        self.native_recorder = StreamlitAudioRecorder()
        self.web_recorder = WebAudioRecorder()
        self.recording_method = self._determine_recording_method()
    
    def _determine_recording_method(self) -> str:
        """Determine the best available recording method"""
        if self.native_recorder.audio_available:
            return "native"
        else:
            return "web"
    
    def render_recording_interface(self) -> Dict[str, Any]:
        """Render the recording interface and return results"""
        result = {
            'audio_data': None,
            'is_recording': False,
            'recording_level': 0.0,
            'method': self.recording_method
        }
        
        if self.recording_method == "native":
            result.update(self._render_native_interface())
        else:
            result.update(self._render_web_interface())
        
        return result
    
    def _render_native_interface(self) -> Dict[str, Any]:
        """Render native PyAudio interface"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Recording status
            if 'recording_state' not in st.session_state:
                st.session_state.recording_state = False
            
            # Record button
            if not st.session_state.recording_state:
                if st.button("🎙️ Start Recording", key="start_record", use_container_width=True):
                    if self.native_recorder.start_recording():
                        st.session_state.recording_state = True
                        st.rerun()
            else:
                if st.button("⏹️ Stop Recording", key="stop_record", use_container_width=True):
                    audio_data = self.native_recorder.stop_recording()
                    st.session_state.recording_state = False
                    st.session_state.recorded_audio = audio_data
                    st.rerun()
        
        # Show recording status
        if st.session_state.recording_state:
            st.markdown("""
            <div class="status-indicator status-error">
                <span>🔴</span>
                <span>Recording in progress...</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Show recording level
            level = self.native_recorder.get_recording_level()
            st.progress(level)
            
            # Auto-refresh to update level
            time.sleep(0.1)
            st.rerun()
        
        # Return recorded audio if available
        audio_data = getattr(st.session_state, 'recorded_audio', None)
        if audio_data is not None:
            st.session_state.recorded_audio = None  # Clear after use
        
        return {
            'audio_data': audio_data,
            'is_recording': st.session_state.recording_state,
            'recording_level': self.native_recorder.get_recording_level() if st.session_state.recording_state else 0.0
        }
    
    def _render_web_interface(self) -> Dict[str, Any]:
        """Render web-based interface"""
        audio_data = self.web_recorder.render_recorder_ui()
        
        return {
            'audio_data': audio_data,
            'is_recording': False,
            'recording_level': 0.0
        }

# Utility functions for audio processing
def save_audio_to_temp_file(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Save audio data to temporary file"""
    try:
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving audio to temp file: {e}")
        return None

def convert_bytes_to_audio(audio_bytes: bytes) -> Optional[np.ndarray]:
    """Convert audio bytes to numpy array"""
    try:
        import soundfile as sf
        import io
        
        # Try to read as audio file
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        return audio_data
        
    except Exception as e:
        logger.error(f"Error converting bytes to audio: {e}")
        # Fallback: assume raw PCM data
        try:
            return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e2:
            logger.error(f"Error with fallback conversion: {e2}")
            return None