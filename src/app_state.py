"""
Application state management for the Emotion-Aware Voice Bot
"""
import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class AppState:
    """Centralized application state management"""
    
    def __init__(self):
        """Initialize application state"""
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        
        # Core application state
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = False
        
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = None
        
        # Recording state
        if 'recording_state' not in st.session_state:
            st.session_state.recording_state = False
        
        if 'recorded_audio' not in st.session_state:
            st.session_state.recorded_audio = None
        
        if 'processing_state' not in st.session_state:
            st.session_state.processing_state = 'idle'  # idle, processing, completed, error
        
        # Conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        # Current session data
        if 'current_session' not in st.session_state:
            st.session_state.current_session = {
                'transcription': '',
                'emotion': '',
                'confidence': 0.0,
                'response': '',
                'processing_time': 0.0,
                'timestamp': None
            }
        
        # UI state
        if 'selected_tab' not in st.session_state:
            st.session_state.selected_tab = 'record'
        
        if 'show_advanced' not in st.session_state:
            st.session_state.show_advanced = False
        
        # Settings
        if 'app_settings' not in st.session_state:
            st.session_state.app_settings = {
                'auto_process': True,
                'show_confidence': True,
                'save_history': True,
                'theme': 'modern',
                'recording_quality': 'high'
            }
        
        # Performance metrics
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {
                'total_sessions': 0,
                'average_processing_time': 0.0,
                'emotion_accuracy': 0.0,
                'user_satisfaction': 0.0
            }
    
    @property
    def is_initialized(self) -> bool:
        """Check if app is initialized"""
        return st.session_state.get('app_initialized', False)
    
    @property
    def pipeline(self):
        """Get the AI pipeline"""
        return st.session_state.get('pipeline', None)
    
    @pipeline.setter
    def pipeline(self, value):
        """Set the AI pipeline"""
        st.session_state.pipeline = value
        st.session_state.app_initialized = True
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return st.session_state.recording_state
    
    @is_recording.setter
    def is_recording(self, value: bool):
        """Set recording state"""
        st.session_state.recording_state = value
    
    @property
    def processing_state(self) -> str:
        """Get current processing state"""
        return st.session_state.processing_state
    
    @processing_state.setter
    def processing_state(self, value: str):
        """Set processing state"""
        st.session_state.processing_state = value
    
    @property
    def conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return st.session_state.conversation_history
    
    def add_conversation(self, 
                        transcription: str,
                        emotion: str,
                        confidence: float,
                        response: str,
                        processing_time: float,
                        audio_duration: float = 0.0,
                        metadata: Optional[Dict] = None):
        """Add a conversation to history"""
        
        conversation_entry = {
            'id': len(st.session_state.conversation_history) + 1,
            'timestamp': datetime.now().isoformat(),
            'transcription': transcription,
            'emotion': emotion,
            'confidence': confidence,
            'response': response,
            'processing_time': processing_time,
            'audio_duration': audio_duration,
            'metadata': metadata or {}
        }
        
        st.session_state.conversation_history.append(conversation_entry)
        
        # Update performance metrics
        self._update_performance_metrics(processing_time)
        
        logger.info(f"Added conversation entry: {emotion} ({confidence:.2f})")
    
    def clear_conversation_history(self):
        """Clear all conversation history"""
        st.session_state.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        history = st.session_state.conversation_history
        
        if not history:
            return {
                'total_conversations': 0,
                'emotions': {},
                'average_confidence': 0.0,
                'average_processing_time': 0.0,
                'total_audio_time': 0.0
            }
        
        # Calculate statistics
        emotions = {}
        total_confidence = 0.0
        total_processing_time = 0.0
        total_audio_time = 0.0
        
        for conv in history:
            emotion = conv.get('emotion', 'unknown')
            emotions[emotion] = emotions.get(emotion, 0) + 1
            total_confidence += conv.get('confidence', 0.0)
            total_processing_time += conv.get('processing_time', 0.0)
            total_audio_time += conv.get('audio_duration', 0.0)
        
        return {
            'total_conversations': len(history),
            'emotions': emotions,
            'average_confidence': total_confidence / len(history),
            'average_processing_time': total_processing_time / len(history),
            'total_audio_time': total_audio_time,
            'most_common_emotion': max(emotions, key=emotions.get) if emotions else 'none'
        }
    
    def update_current_session(self, **kwargs):
        """Update current session data"""
        for key, value in kwargs.items():
            if key in st.session_state.current_session:
                st.session_state.current_session[key] = value
    
    def get_current_session(self) -> Dict[str, Any]:
        """Get current session data"""
        return st.session_state.current_session.copy()
    
    def reset_current_session(self):
        """Reset current session data"""
        st.session_state.current_session = {
            'transcription': '',
            'emotion': '',
            'confidence': 0.0,
            'response': '',
            'processing_time': 0.0,
            'timestamp': None
        }
    
    def get_setting(self, key: str, default=None):
        """Get application setting"""
        return st.session_state.app_settings.get(key, default)
    
    def set_setting(self, key: str, value):
        """Set application setting"""
        st.session_state.app_settings[key] = value
    
    def export_conversation_history(self) -> str:
        """Export conversation history as JSON"""
        try:
            return json.dumps(st.session_state.conversation_history, indent=2)
        except Exception as e:
            logger.error(f"Error exporting conversation history: {e}")
            return "{}"
    
    def import_conversation_history(self, json_data: str) -> bool:
        """Import conversation history from JSON"""
        try:
            imported_data = json.loads(json_data)
            if isinstance(imported_data, list):
                st.session_state.conversation_history = imported_data
                return True
            else:
                logger.error("Invalid conversation history format")
                return False
        except Exception as e:
            logger.error(f"Error importing conversation history: {e}")
            return False
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        metrics = st.session_state.performance_metrics
        
        # Update total sessions
        metrics['total_sessions'] += 1
        
        # Update average processing time
        current_avg = metrics['average_processing_time']
        total_sessions = metrics['total_sessions']
        metrics['average_processing_time'] = (
            (current_avg * (total_sessions - 1) + processing_time) / total_sessions
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return st.session_state.performance_metrics.copy()

class UIState:
    """UI-specific state management"""
    
    @staticmethod
    def set_tab(tab_name: str):
        """Set active tab"""
        st.session_state.selected_tab = tab_name
    
    @staticmethod
    def get_tab() -> str:
        """Get active tab"""
        return st.session_state.get('selected_tab', 'record')
    
    @staticmethod
    def toggle_advanced():
        """Toggle advanced settings"""
        st.session_state.show_advanced = not st.session_state.get('show_advanced', False)
    
    @staticmethod
    def is_advanced_shown() -> bool:
        """Check if advanced settings are shown"""
        return st.session_state.get('show_advanced', False)
    
    @staticmethod
    def set_processing_status(status: str, message: str = ""):
        """Set processing status for UI feedback"""
        st.session_state.processing_status = {
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def get_processing_status() -> Dict[str, str]:
        """Get processing status"""
        return st.session_state.get('processing_status', {
            'status': 'idle',
            'message': '',
            'timestamp': ''
        })

# Global app state instance
app_state = AppState()
ui_state = UIState()