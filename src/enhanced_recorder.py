"""
Enhanced audio recorder with multiple fallback options for maximum compatibility
"""
import streamlit as st
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, Callable
import tempfile
import os

logger = logging.getLogger(__name__)

class EnhancedAudioRecorder:
    """Enhanced audio recorder with multiple recording methods"""
    
    def __init__(self):
        """Initialize enhanced audio recorder"""
        self.recording_methods = self._detect_available_methods()
        self.current_method = self._select_best_method()
        self.is_recording = False
        self.recorded_audio = None
    
    def _detect_available_methods(self) -> Dict[str, bool]:
        """Detect available recording methods"""
        methods = {
            'streamlit_webrtc': False,
            'audio_recorder_streamlit': False,
            'pyaudio': False,
            'file_upload': True  # Always available
        }
        
        # Check streamlit-webrtc
        try:
            import streamlit_webrtc
            methods['streamlit_webrtc'] = True
            logger.info("streamlit-webrtc available")
        except ImportError:
            pass
        
        # Check audio-recorder-streamlit
        try:
            import audio_recorder_streamlit
            methods['audio_recorder_streamlit'] = True
            logger.info("audio-recorder-streamlit available")
        except ImportError:
            pass
        
        # Check PyAudio
        try:
            import pyaudio
            methods['pyaudio'] = True
            logger.info("PyAudio available")
        except ImportError:
            pass
        
        return methods
    
    def _select_best_method(self) -> str:
        """Select the best available recording method"""
        # Priority order
        priority = [
            'audio_recorder_streamlit',
            'streamlit_webrtc', 
            'pyaudio',
            'file_upload'
        ]
        
        for method in priority:
            if self.recording_methods.get(method, False):
                logger.info(f"Selected recording method: {method}")
                return method
        
        return 'file_upload'
    
    def render_recording_interface(self) -> Dict[str, Any]:
        """Render the recording interface with both recording and upload options"""
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                🎙️ Voice Input Options
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different input methods
        if self.current_method != 'file_upload':
            tab1, tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload File"])
            
            with tab1:
                # Show available recording method
                method_info = {
                    'audio_recorder_streamlit': '🎙️ Real-time browser recording',
                    'streamlit_webrtc': '📡 WebRTC real-time recording',
                    'pyaudio': '🖥️ System audio recording'
                }
                
                st.info(f"Recording method: {method_info.get(self.current_method, 'Unknown')}")
                
                # Render recording interface
                if self.current_method == 'audio_recorder_streamlit':
                    recording_result = self._render_audio_recorder_streamlit()
                elif self.current_method == 'streamlit_webrtc':
                    recording_result = self._render_streamlit_webrtc()
                elif self.current_method == 'pyaudio':
                    recording_result = self._render_pyaudio()
                else:
                    recording_result = {'audio_data': None, 'method': self.current_method, 'success': False}
                
                # Return recording result if we got audio
                if recording_result.get('audio_data') is not None:
                    return recording_result
            
            with tab2:
                # Always show file upload as alternative
                upload_result = self._render_file_upload()
                if upload_result.get('audio_data') is not None:
                    return upload_result
            
            # Return empty result if no audio from either tab
            return {'audio_data': None, 'method': 'none', 'success': False}
        
        else:
            # Only file upload available
            return self._render_file_upload()
    
    def _render_audio_recorder_streamlit(self) -> Dict[str, Any]:
        """Render audio-recorder-streamlit interface"""
        try:
            from audio_recorder_streamlit import audio_recorder
            
            st.markdown("### 🎙️ Click to Record")
            
            # Audio recorder component
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=2.0,
                sample_rate=16000
            )
            
            if audio_bytes:
                st.success("✅ Audio recorded successfully!")
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Show audio player
                st.audio(audio_bytes, format='audio/wav')
                
                return {
                    'audio_data': audio_array,
                    'method': 'audio_recorder_streamlit',
                    'success': True
                }
            
            return {'audio_data': None, 'method': 'audio_recorder_streamlit', 'success': False}
            
        except Exception as e:
            logger.error(f"Error with audio-recorder-streamlit: {e}")
            return self._render_file_upload()
    
    def _render_streamlit_webrtc(self) -> Dict[str, Any]:
        """Render streamlit-webrtc interface"""
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
            import av
            import queue
            
            st.markdown("### 📡 WebRTC Recording")
            
            # WebRTC configuration
            rtc_configuration = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })
            
            # Create audio queue
            if 'audio_frames' not in st.session_state:
                st.session_state.audio_frames = queue.Queue()
            
            def audio_frame_callback(frame):
                """Process audio frames"""
                sound = frame.to_ndarray()
                st.session_state.audio_frames.put(sound)
                return frame
            
            # WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="audio-recorder-webrtc",
                mode=WebRtcMode.SENDONLY,
                audio_frame_callback=audio_frame_callback,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": False, "audio": True},
            )
            
            # Process recorded audio
            if not webrtc_ctx.state.playing and st.session_state.audio_frames.qsize() > 0:
                # Collect all audio frames
                audio_data = []
                while not st.session_state.audio_frames.empty():
                    try:
                        frame = st.session_state.audio_frames.get_nowait()
                        audio_data.append(frame)
                    except queue.Empty:
                        break
                
                if audio_data:
                    # Combine audio frames
                    combined_audio = np.concatenate(audio_data, axis=0)
                    if len(combined_audio.shape) > 1:
                        combined_audio = np.mean(combined_audio, axis=1)
                    
                    st.success("✅ Audio recorded via WebRTC!")
                    
                    return {
                        'audio_data': combined_audio,
                        'method': 'streamlit_webrtc',
                        'success': True
                    }
            
            return {'audio_data': None, 'method': 'streamlit_webrtc', 'success': False}
            
        except Exception as e:
            logger.error(f"Error with streamlit-webrtc: {e}")
            return self._render_file_upload()
    
    def _render_pyaudio(self) -> Dict[str, Any]:
        """Render PyAudio interface"""
        try:
            import pyaudio
            
            st.markdown("### 🖥️ System Audio Recording")
            st.warning("⚠️ PyAudio recording requires additional setup and may not work in all environments.")
            
            # Simple recording interface
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎙️ Start Recording", key="pyaudio_start"):
                    st.session_state.pyaudio_recording = True
                    st.rerun()
            
            with col2:
                if st.button("⏹️ Stop Recording", key="pyaudio_stop"):
                    st.session_state.pyaudio_recording = False
                    st.rerun()
            
            # Recording status
            if st.session_state.get('pyaudio_recording', False):
                st.error("🔴 Recording... (This is a placeholder - real implementation would record audio)")
                
                # Placeholder for actual recording
                # In a real implementation, this would use PyAudio to record
                time.sleep(0.1)
                st.rerun()
            
            # Fallback to file upload
            st.info("💡 For now, please use file upload below:")
            return self._render_file_upload()
            
        except Exception as e:
            logger.error(f"Error with PyAudio: {e}")
            return self._render_file_upload()
    
    def _render_file_upload(self) -> Dict[str, Any]:
        """Render file upload interface"""
        st.markdown("### 📁 Upload Audio File")
        st.info("💡 You can also upload audio files if real-time recording isn't working")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
            help="Upload an audio file to analyze emotions and generate responses"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Show file info
            st.write(f"**File size:** {uploaded_file.size:,} bytes")
            
            # Audio player
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format='audio/wav')
            
            # Convert to numpy array
            try:
                import soundfile as sf
                import io
                
                # Reset file pointer
                uploaded_file.seek(0)
                audio_bytes = uploaded_file.read()
                
                # Try to read the audio file
                audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
                
                st.success(f"✅ Audio loaded: {len(audio_array)} samples at {sample_rate}Hz")
                
                # Ensure mono
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)
                    st.info("🔄 Converted stereo to mono")
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                    st.info(f"🔄 Resampled from {sample_rate}Hz to 16000Hz")
                
                # Basic validation
                final_sample_rate = 16000  # After resampling
                duration = len(audio_array) / final_sample_rate
                rms = np.sqrt(np.mean(audio_array**2))
                
                st.info(f"📊 Final duration: {duration:.2f}s, RMS level: {rms:.3f}")
                
                if duration < 0.1:
                    st.warning("⚠️ Audio is very short, transcription may not work well")
                elif duration > 30:
                    st.warning("⚠️ Audio is long, processing may take time")
                
                if rms < 0.001:
                    st.warning("⚠️ Audio level is very low, transcription may not work well")
                
                return {
                    'audio_data': audio_array,
                    'method': 'file_upload',
                    'success': True,
                    'filename': uploaded_file.name,
                    'duration': duration,
                    'sample_rate': 16000
                }
                
            except Exception as e:
                st.error(f"❌ Error processing audio file: {e}")
                st.error("💡 Try uploading a different audio file (WAV, MP3, M4A)")
                return {'audio_data': None, 'method': 'file_upload', 'success': False}
        
        return {'audio_data': None, 'method': 'file_upload', 'success': False}
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about available recording methods"""
        return {
            'current_method': self.current_method,
            'available_methods': self.recording_methods,
            'method_descriptions': {
                'audio_recorder_streamlit': 'Browser-based real-time recording',
                'streamlit_webrtc': 'WebRTC real-time recording',
                'pyaudio': 'System-level audio recording',
                'file_upload': 'File upload (always available)'
            }
        }

class RecordingInstructions:
    """Component to show recording instructions and tips"""
    
    @staticmethod
    def render_instructions():
        """Render recording instructions"""
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                💡 Recording Tips
            </div>
            <div style="padding: 1rem 0;">
                <ul style="margin: 0; padding-left: 1.5rem;">
                    <li><strong>Speak clearly</strong> and at normal volume</li>
                    <li><strong>Record for 2-10 seconds</strong> for best results</li>
                    <li><strong>Minimize background noise</strong> when possible</li>
                    <li><strong>Express emotions naturally</strong> in your voice</li>
                    <li><strong>Wait for processing</strong> to complete before recording again</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_troubleshooting():
        """Render troubleshooting information"""
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **If recording doesn't work:**
            
            1. **Check browser permissions** - Allow microphone access
            2. **Try refreshing the page** - Sometimes helps with WebRTC
            3. **Use file upload** - Always works as a fallback
            4. **Check your microphone** - Test in other applications
            5. **Try a different browser** - Chrome/Edge work best
            
            **Supported audio formats:**
            - WAV (recommended)
            - MP3
            - M4A
            - OGG
            - FLAC
            
            **For best results:**
            - Use a quiet environment
            - Speak directly into the microphone
            - Keep recordings between 2-10 seconds
            - Express emotions clearly in your voice
            """)

def create_enhanced_recorder() -> EnhancedAudioRecorder:
    """Factory function to create enhanced audio recorder"""
    return EnhancedAudioRecorder()