4. Ollama with LLaMA-2/Mistral (Response Generation)
Overview
Ollama allows you to run large language models (LLMs) locally on your machine. It provides an API interface similar to OpenAI's, making it perfect for generating empathetic responses based on detected emotions from voice input.

Installation
1. Install Ollama
bash
# On macOS
brew install ollama

# On Linux
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows
# Download from https://ollama.ai/download
2. Install Python Client
bash
pip install ollama
3. Pull Models
bash
# Pull LLaMA 2 (7B parameter model - good balance of speed/quality)
ollama pull llama2

# Pull Mistral (7B parameter model - excellent for instruction following)
ollama pull mistral

# Pull Mixtral (Mixture of Experts - higher quality but more resource-intensive)
ollama pull mixtral

# List available models
ollama list
Basic Usage
Simple Text Generation
python
import ollama

# Basic text generation
response = ollama.generate(model='llama2', prompt='Why is the sky blue?')
print(response['response'])
Chat Completion Interface
python
def simple_chat(prompt, model='llama2'):
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

# Example usage
response = simple_chat("Hello, how are you?")
print(response)
Emotion-Aware Response Generation
Core Response Generation Function
python
import ollama
import json
from typing import Dict, List

class EmotionAwareResponder:
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        self.conversation_history = []
        
    def generate_empathetic_response(self, transcription: str, emotion: str, 
                                   emotion_confidence: float = 0.8) -> str:
        """
        Generate empathetic response based on speech content and detected emotion
        """
        # System prompt to guide the model's behavior
        system_prompt = f"""You are an empathetic AI assistant that responds to users based on their emotional state.
        
        Current detected emotion: {emotion} (confidence: {emotion_confidence:.2f})
        
        Guidelines:
        1. Acknowledge and validate the user's emotion
        2. Respond appropriately to the emotional context
        3. Maintain a supportive and understanding tone
        4. Keep responses concise (1-2 sentences)
        5. Adapt your language to match the emotion:
           - Happy/Excited: Be enthusiastic and positive
           - Sad: Be comforting and supportive
           - Angry: Be calming and understanding
           - Neutral: Be professional and helpful
           - Fearful: Be reassuring and confident
        
        User's transcription: "{transcription}"
        """
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user', 
                        'content': f"I'm feeling {emotion}. {transcription}"
                    }
                ],
                options={
                    'temperature': 0.7,  # Controls creativity (0.0-1.0)
                    'top_p': 0.9,       # Nucleus sampling parameter
                    'top_k': 40,        # Top-k sampling parameter
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self.get_fallback_response(emotion)

    def get_fallback_response(self, emotion: str) -> str:
        """Fallback responses if LLM fails"""
        fallback_responses = {
            'happy': "I'm glad to hear you're feeling positive! How can I help you today?",
            'sad': "I'm sorry you're feeling down. I'm here to listen and help however I can.",
            'angry': "I understand you're feeling frustrated. Let's work through this together.",
            'neutral': "Thank you for sharing. How can I assist you today?",
            'excited': "That's exciting! I'd love to hear more about what's got you so energized!",
            'calm': "It's wonderful that you're feeling peaceful. How can I support you?",
            'fearful': "I sense some concern. Remember, I'm here to help you feel more secure."
        }
        return fallback_responses.get(emotion, "Thank you for sharing. How can I help you?")

# Example usage
responder = EmotionAwareResponder(model_name="mistral")
response = responder.generate_empathetic_response(
    transcription="I just got some great news about my project",
    emotion="excited",
    emotion_confidence=0.92
)
print(response)
Conversation History Management
python
class ConversationalResponder(EmotionAwareResponder):
    def __init__(self, model_name: str = "llama2", max_history: int = 10):
        super().__init__(model_name)
        self.max_history = max_history
        self.conversation_history = []
    
    def add_to_history(self, role: str, content: str, emotion: str = None):
        """Add message to conversation history"""
        message = {
            'role': role,
            'content': content,
            'emotion': emotion,
            'timestamp': time.time()
        }
        self.conversation_history.append(message)
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def generate_contextual_response(self, transcription: str, emotion: str, 
                                  emotion_confidence: float = 0.8) -> str:
        """
        Generate response with conversation context
        """
        # Build messages with history
        messages = [
            {
                'role': 'system',
                'content': f"""You are an empathetic AI assistant. Respond appropriately to the user's emotion.
                
                Detected emotion: {emotion} (confidence: {emotion_confidence:.2f})
                Be supportive, understanding, and context-aware."""
            }
        ]
        
        # Add conversation history
        for msg in self.conversation_history[-4:]:  # Last 4 messages for context
            messages.append({'role': msg['role'], 'content': msg['content']})
        
        # Add current message
        current_message = f"I'm feeling {emotion}. {transcription}"
        messages.append({'role': 'user', 'content': current_message})
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': max(0.3, min(0.9, emotion_confidence)),  # Adjust based on confidence
                    'top_p': 0.9,
                }
            )
            
            response_text = response['message']['content']
            
            # Update history
            self.add_to_history('user', transcription, emotion)
            self.add_to_history('assistant', response_text)
            
            return response_text
            
        except Exception as e:
            print(f"Error in contextual response: {e}")
            return self.get_fallback_response(emotion)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
Advanced Response Generation with Custom Parameters
python
def generate_custom_response(transcription: str, emotion: str, 
                           model_params: Dict = None) -> str:
    """
    Generate response with custom model parameters
    """
    default_params = {
        'model': 'llama2',
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 40,
        'num_predict': 150,  # Max tokens to generate
        'repeat_penalty': 1.1,
        'stop': ['\n\n', 'User:', 'Assistant:']
    }
    
    # Merge with custom parameters
    if model_params:
        default_params.update(model_params)
    
    prompt = f"""<s>[INST] <<SYS>>
You are an empathetic AI assistant. The user is feeling {emotion}.
Respond with understanding and support. Keep your response concise.
<</SYS>>

User: {transcription}
Assistant: [/INST]"""
    
    try:
        response = ollama.generate(
            model=default_params['model'],
            prompt=prompt,
            options={
                'temperature': default_params['temperature'],
                'top_p': default_params['top_p'],
                'top_k': default_params['top_k'],
                'num_predict': default_params['num_predict'],
                'repeat_penalty': default_params['repeat_penalty'],
                'stop': default_params['stop']
            }
        )
        return response['response']
    except Exception as e:
        print(f"Error in custom response generation: {e}")
        return f"I understand you're feeling {emotion}. How can I help?"
Emotion-Specific Response Templates
python
class EmotionResponseTemplates:
    @staticmethod
    def get_emotion_template(emotion: str, transcription: str) -> str:
        """Get emotion-specific prompt template"""
        templates = {
            'happy': f"""<s>[INST] <<SYS>>
You're speaking with someone who is happy and excited. Match their positive energy!
Be enthusiastic, celebratory, and encouraging. Keep it brief and genuine.
<</SYS>>

User: {transcription}
Assistant: [/INST]""",
            
            'sad': f"""<s>[INST] <<SYS>>
You're speaking with someone who is sad. Be comforting, supportive, and understanding.
Offer gentle encouragement without being overly cheerful. Show empathy.
<</SYS>>

User: {transcription}
Assistant: [/INST]""",
            
            'angry': f"""<s>[INST] <<SYS>>
You're speaking with someone who is angry. Stay calm, validate their feelings, and be solution-oriented.
Don't escalate; instead, help them feel heard and understood.
<</SYS>>

User: {transcription}
Assistant: [/INST]""",
            
            'neutral': f"""<s>[INST] <<SYS>>
You're speaking with someone in a neutral state. Be professional, helpful, and clear.
Provide useful information while maintaining a friendly tone.
<</SYS>>

User: {transcription}
Assistant: [/INST]"""
        }
        return templates.get(emotion, templates['neutral'])
    
    def generate_templated_response(self, transcription: str, emotion: str) -> str:
        """Generate response using emotion-specific template"""
        prompt = self.get_emotion_template(emotion, transcription)
        
        try:
            response = ollama.generate(
                model='mistral',  # Mistral works well with instruction templates
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'num_predict': 100
                }
            )
            return response['response']
        except Exception as e:
            print(f"Template response error: {e}")
            return f"I understand you're feeling {emotion}. Thank you for sharing."
Batch Processing for Multiple Emotions
python
def process_emotional_dialogue(conversation_data: List[Dict]) -> List[Dict]:
    """
    Process a conversation with multiple emotional turns
    """
    responses = []
    responder = ConversationalResponder(model_name="llama2")
    
    for turn in conversation_data:
        transcription = turn.get('text', '')
        emotion = turn.get('emotion', 'neutral')
        confidence = turn.get('confidence', 0.8)
        
        response = responder.generate_contextual_response(
            transcription, emotion, confidence
        )
        
        responses.append({
            'user_input': transcription,
            'detected_emotion': emotion,
            'confidence': confidence,
            'ai_response': response,
            'timestamp': time.time()
        })
    
    return responses

# Example batch processing
conversation = [
    {'text': 'I had a really tough day at work', 'emotion': 'sad', 'confidence': 0.85},
    {'text': 'But then I got some good news', 'emotion': 'happy', 'confidence': 0.78},
    {'text': 'Now I feel much better', 'emotion': 'calm', 'confidence': 0.92}
]

results = process_emotional_dialogue(conversation)
for result in results:
    print(f"Emotion: {result['detected_emotion']}")
    print(f"User: {result['user_input']}")
    print(f"AI: {result['ai_response']}")
    print("---")
Error Handling and Model Management
python
class RobustOllamaClient:
    def __init__(self, primary_model: str = "llama2", fallback_model: str = "mistral"):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.available_models = self._get_available_models()
    
    def _get_available_models(self) -> List[str]:
        """Check which models are available locally"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            models = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip():
                    models.append(line.split()[0])
            return models
        except:
            return []
    
    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a model is available, pull if necessary"""
        if model_name in self.available_models:
            return True
        
        try:
            print(f"Pulling model {model_name}...")
            subprocess.run(['ollama', 'pull', model_name], check=True)
            self.available_models.append(model_name)
            return True
        except Exception as e:
            print(f"Failed to pull model {model_name}: {e}")
            return False
    
    def robust_generate(self, prompt: str, **kwargs) -> str:
        """Generate response with fallback logic"""
        model = kwargs.get('model', self.primary_model)
        
        if not self.ensure_model_available(model):
            print(f"Primary model {model} not available, trying fallback...")
            model = self.fallback_model
            if not self.ensure_model_available(model):
                return "I'm currently unavailable. Please try again later."
        
        try:
            response = ollama.generate(model=model, prompt=prompt, **kwargs)
            return response['response']
        except Exception as e:
            print(f"Generation failed: {e}")
            return "I apologize, I'm having trouble responding right now."
Integration with Emotion Pipeline
python
def complete_emotion_pipeline(audio_path: str, model_name: str = "mistral") -> Dict:
    """
    Complete pipeline: Audio → Transcription → Emotion → Response
    """
    # This would integrate with your previous components
    # For demonstration, we'll use placeholder functions
    
    # 1. Transcribe audio (using Whisper)
    transcription = "I'm feeling really good about today's progress"
    
    # 2. Detect emotion (using pyAudioAnalysis)
    emotion = "happy"
    confidence = 0.88
    
    # 3. Generate response
    responder = EmotionAwareResponder(model_name=model_name)
    response = responder.generate_empathetic_response(transcription, emotion, confidence)
    
    return {
        'transcription': transcription,
        'detected_emotion': emotion,
        'confidence': confidence,
        'response': response,
        'timestamp': time.time()
    }

# Run complete pipeline
result = complete_emotion_pipeline("audio.wav")
print(f"Transcription: {result['transcription']}")
print(f"Emotion: {result['detected_emotion']} ({result['confidence']:.2f})")
print(f"Response: {result['response']}")
This comprehensive Ollama integration provides everything you need for emotion-aware response generation, including conversation history management, error handling, and integration with your emotion detection pipeline.

