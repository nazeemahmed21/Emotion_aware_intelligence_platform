"""
LLM response generation using Ollama for emotion-aware responses
"""
import requests
import json
import logging
from typing import Dict, List, Optional, Union
import time
import subprocess
import sys

from config import LLM_CONFIG

logger = logging.getLogger(__name__)

class OllamaResponseGenerator:
    """Generate empathetic responses using Ollama LLM"""
    
    def __init__(self, 
                 model_name: str = LLM_CONFIG["model_name"],
                 base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama response generator
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.fallback_model = LLM_CONFIG["fallback_model"]
        self.base_url = base_url
        self.conversation_history = []
        self.max_history = 10
        
        # Check if Ollama is running and models are available
        self._check_ollama_status()
    
    def _check_ollama_status(self) -> bool:
        """Check if Ollama is running and models are available"""
        try:
            # Check if Ollama server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                # Check for exact match or with :latest suffix
                available_models = []
                for model in model_names:
                    base_name = model.split(':')[0]  # Remove :latest suffix
                    available_models.append(base_name)
                
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                    if self.fallback_model in available_models:
                        logger.info(f"Using fallback model: {self.fallback_model}")
                        self.model_name = self.fallback_model
                    else:
                        # Try to find the actual model name with suffix
                        for model in model_names:
                            if model.startswith(self.model_name):
                                logger.info(f"Found model with suffix: {model}")
                                self.model_name = model
                                break
                            elif model.startswith(self.fallback_model):
                                logger.info(f"Found fallback model with suffix: {model}")
                                self.model_name = model
                                break
                        else:
                            logger.error("No suitable models found")
                            return False
                
                logger.info(f"Ollama is running with model: {self.model_name}")
                return True
            else:
                logger.error(f"Ollama server responded with status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama server: {e}")
            logger.info("Make sure Ollama is installed and running: 'ollama serve'")
            return False
    
    def _ensure_model_available(self, model_name: str) -> bool:
        """Ensure model is pulled and available"""
        try:
            # Try to pull the model if not available
            logger.info(f"Ensuring model {model_name} is available...")
            result = subprocess.run(
                ['ollama', 'pull', model_name], 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Model {model_name} is ready")
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while pulling model {model_name}")
            return False
        except FileNotFoundError:
            logger.error("Ollama CLI not found. Please install Ollama first.")
            return False
        except Exception as e:
            logger.error(f"Error ensuring model availability: {e}")
            return False
    
    def generate_empathetic_response(self, 
                                   transcription: str, 
                                   emotion: str, 
                                   confidence: float = 0.8,
                                   context: Optional[str] = None) -> Dict[str, Union[str, float]]:
        """
        Generate empathetic response based on emotion and transcription
        
        Args:
            transcription: User's speech transcription
            emotion: Detected emotion
            confidence: Emotion detection confidence
            context: Additional context for response generation
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Create emotion-aware system prompt
            system_prompt = self._create_system_prompt(emotion, confidence)
            
            # Create user message
            user_message = self._create_user_message(transcription, emotion, context)
            
            # Generate response
            response = self._call_ollama_api(system_prompt, user_message)
            
            if response:
                # Add to conversation history
                self._add_to_history('user', transcription, emotion)
                self._add_to_history('assistant', response)
                
                return {
                    'response': response,
                    'emotion': emotion,
                    'confidence': confidence,
                    'model': self.model_name,
                    'timestamp': time.time()
                }
            else:
                # Fallback response
                return self._get_fallback_response(emotion, transcription)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(emotion, transcription)
    
    def _create_system_prompt(self, emotion: str, confidence: float) -> str:
        """Create emotion-aware system prompt"""
        base_prompt = """You are an empathetic AI assistant that responds to users based on their emotional state. 
        Your responses should be supportive, understanding, and appropriate to the detected emotion.
        Keep responses concise (1-2 sentences) and natural."""
        
        emotion_guidelines = {
            'happy': "The user is feeling happy and positive. Match their energy with enthusiasm and encouragement. Celebrate their positive mood.",
            'sad': "The user is feeling sad or down. Be gentle, comforting, and supportive. Offer understanding and hope without being overly cheerful.",
            'angry': "The user is feeling angry or frustrated. Stay calm and understanding. Validate their feelings and help them feel heard.",
            'neutral': "The user is in a neutral emotional state. Be helpful, professional, and friendly without being overly emotional.",
            'excited': "The user is feeling excited and energetic. Match their enthusiasm and encourage their positive energy.",
            'calm': "The user is feeling calm and peaceful. Maintain a serene, supportive tone that matches their tranquil state.",
            'fearful': "The user is feeling anxious or fearful. Be reassuring, confident, and supportive. Help them feel safe and understood."
        }
        
        emotion_guide = emotion_guidelines.get(emotion, emotion_guidelines['neutral'])
        confidence_note = f"Emotion detection confidence: {confidence:.1%}"
        
        return f"{base_prompt}\n\n{emotion_guide}\n\n{confidence_note}"
    
    def _create_user_message(self, transcription: str, emotion: str, context: Optional[str] = None) -> str:
        """Create user message with emotion context"""
        message = f"I'm feeling {emotion}. {transcription}"
        
        if context:
            message += f"\n\nAdditional context: {context}"
        
        return message
    
    def _call_ollama_api(self, system_prompt: str, user_message: str) -> Optional[str]:
        """Call Ollama API to generate response"""
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history (last few messages)
            recent_history = self.conversation_history[-4:]  # Last 4 messages
            messages.extend(recent_history)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # API request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "options": {
                    "temperature": LLM_CONFIG["temperature"],
                    "num_predict": LLM_CONFIG["max_tokens"],
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                },
                "stream": False
            }
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=LLM_CONFIG["timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Ollama API timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API call: {e}")
            return None
    
    def _add_to_history(self, role: str, content: str, emotion: Optional[str] = None):
        """Add message to conversation history"""
        message = {
            "role": role,
            "content": content
        }
        
        if emotion:
            message["emotion"] = emotion
        
        self.conversation_history.append(message)
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def _get_fallback_response(self, emotion: str, transcription: str) -> Dict[str, Union[str, float]]:
        """Generate fallback response when LLM is unavailable"""
        fallback_responses = {
            'happy': [
                "That's wonderful to hear! I'm glad you're feeling positive.",
                "It's great that you're in such a good mood! How can I help you today?",
                "Your happiness is contagious! What's bringing you joy?"
            ],
            'sad': [
                "I'm sorry you're feeling down. I'm here to listen and support you.",
                "It sounds like you're going through a tough time. How can I help?",
                "I understand you're feeling sad. Remember that it's okay to feel this way."
            ],
            'angry': [
                "I can hear that you're frustrated. Let's work through this together.",
                "I understand you're feeling angry. Your feelings are valid.",
                "It sounds like something is really bothering you. I'm here to help."
            ],
            'neutral': [
                "Thank you for sharing. How can I assist you today?",
                "I'm here to help. What would you like to talk about?",
                "How can I support you right now?"
            ],
            'excited': [
                "I can feel your excitement! That's amazing!",
                "Your energy is wonderful! Tell me more about what's got you so excited.",
                "That's fantastic! I love your enthusiasm."
            ],
            'calm': [
                "It's nice to hear you're feeling peaceful. How can I help you today?",
                "Your calm energy is lovely. What's on your mind?",
                "I appreciate your tranquil state. How can I support you?"
            ],
            'fearful': [
                "I understand you're feeling anxious. You're safe here, and I'm here to help.",
                "It's okay to feel worried sometimes. Let's work through this together.",
                "I hear your concern. Remember that you're not alone in this."
            ]
        }
        
        import random
        responses = fallback_responses.get(emotion, fallback_responses['neutral'])
        selected_response = random.choice(responses)
        
        return {
            'response': selected_response,
            'emotion': emotion,
            'confidence': 0.8,
            'model': 'fallback',
            'timestamp': time.time()
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation"""
        if not self.conversation_history:
            return {'message_count': 0, 'emotions': [], 'duration': 0}
        
        # Count emotions
        emotions = []
        for msg in self.conversation_history:
            if 'emotion' in msg:
                emotions.append(msg['emotion'])
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            'message_count': len(self.conversation_history),
            'emotions': emotion_counts,
            'most_common_emotion': max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
        }

class FallbackResponseGenerator:
    """Simple rule-based response generator as fallback"""
    
    def __init__(self):
        self.response_templates = {
            'happy': [
                "I'm so glad to hear you're feeling happy! {transcription} sounds wonderful.",
                "Your positive energy is amazing! Thanks for sharing that {transcription}.",
                "That's fantastic! It's great to hear about {transcription}."
            ],
            'sad': [
                "I'm sorry you're feeling sad about {transcription}. I'm here for you.",
                "It sounds like {transcription} is really affecting you. That's understandable.",
                "I hear that you're going through something difficult with {transcription}."
            ],
            'angry': [
                "I understand you're frustrated about {transcription}. Your feelings are valid.",
                "It sounds like {transcription} is really bothering you. Let's talk about it.",
                "I can hear your frustration about {transcription}. How can I help?"
            ],
            'neutral': [
                "Thank you for sharing about {transcription}. How can I help you?",
                "I understand you mentioned {transcription}. What would you like to discuss?",
                "Thanks for telling me about {transcription}. What's on your mind?"
            ]
        }
    
    def generate_response(self, transcription: str, emotion: str) -> str:
        """Generate simple template-based response"""
        templates = self.response_templates.get(emotion, self.response_templates['neutral'])
        import random
        template = random.choice(templates)
        
        # Simple keyword extraction for personalization
        keywords = transcription.lower().split()[:3]  # First 3 words
        context = ' '.join(keywords) if keywords else 'what you shared'
        
        return template.format(transcription=context)

# Convenience functions
def generate_response(transcription: str, emotion: str, confidence: float = 0.8) -> str:
    """
    Quick response generation
    
    Args:
        transcription: User's speech
        emotion: Detected emotion
        confidence: Emotion confidence
        
    Returns:
        Generated response text
    """
    generator = OllamaResponseGenerator()
    result = generator.generate_empathetic_response(transcription, emotion, confidence)
    return result.get('response', 'I understand. How can I help you?')

# Global generator instance
_global_generator = None

def get_response_generator() -> OllamaResponseGenerator:
    """
    Get global response generator instance (singleton pattern)
    
    Returns:
        OllamaResponseGenerator instance
    """
    global _global_generator
    if _global_generator is None:
        _global_generator = OllamaResponseGenerator()
    return _global_generator